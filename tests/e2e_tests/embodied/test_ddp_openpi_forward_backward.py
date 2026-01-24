# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test script for DDP actor worker with OpenPi model.
Tests forward and backward pass with mock data.

Usage:
    torchrun --nproc_per_node=1 test_ddp_openpi_forward_backward.py \
        --config-path . --config-name test_ddp_openpi
"""

import os
import sys
import json

import hydra
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

# Add repo path to sys.path
REPO_PATH = os.environ.get("REPO_PATH", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, REPO_PATH)

from rlinf.algorithms.registry import policy_loss
from rlinf.config import SupportedModel


def create_mock_data(batch_size: int, device: torch.device, forward_type: str = "default_forward") -> dict:
    """
    Create mock data matching the shapes from the log:
    - prev_logprobs: (batch, action_chunks=5, action_dim=7)
    - prev_values: (batch, 1)
    - dones: (batch, action_chunks=5)
    - terminations: (batch, action_chunks=5)
    - truncations: (batch, action_chunks=5)
    - rewards: (batch, action_chunks=5)
    - chains: (batch, num_steps+1=5, action_horizon=50, action_dim=32)
    - denoise_inds: (batch, num_steps=4)
    - tokenized_prompt: (batch, 48)
    - tokenized_prompt_mask: (batch, 48)
    - observation/image: (batch, 256, 256, 3)
    - observation/state: (batch, 8)
    - observation/wrist_image: (batch, 256, 256, 3)
    - loss_mask: (batch, 1)
    - loss_mask_sum: (batch, 1)
    - advantages: (batch, 1)
    - returns: (batch, 1)
    
    For awr_forward, also includes:
    - actions: (batch, action_horizon=50, action_dim=32)
    """
    action_chunks = 5
    action_dim = 7
    num_steps = 4
    action_horizon = 50
    action_dim_full = 32  # pi0's internal action dimension
    prompt_len = 48
    img_size = 256
    state_dim = 8

    data = {
        # RL data
        "prev_logprobs": torch.randn(batch_size, action_chunks, action_dim, device=device),
        "prev_values": torch.randn(batch_size, 1, device=device),
        "dones": torch.zeros(batch_size, action_chunks, dtype=torch.bool, device=device),
        "terminations": torch.zeros(batch_size, action_chunks, dtype=torch.bool, device=device),
        "truncations": torch.zeros(batch_size, action_chunks, dtype=torch.bool, device=device),
        "rewards": torch.randn(batch_size, action_chunks, device=device),
        # Flow matching data
        "chains": torch.randn(batch_size, num_steps + 1, action_horizon, action_dim_full, device=device),
        "denoise_inds": torch.randint(0, num_steps, (batch_size, num_steps), device=device),
        # Tokenized prompt
        "tokenized_prompt": torch.randint(0, 1000, (batch_size, prompt_len), device=device),
        "tokenized_prompt_mask": torch.ones(batch_size, prompt_len, dtype=torch.bool, device=device),
        # Observations (HWC format as per OpenPi)
        "observation/image": torch.randint(0, 256, (batch_size, img_size, img_size, 3), dtype=torch.uint8, device=device),
        "observation/state": torch.randn(batch_size, state_dim, device=device, dtype=torch.float32),
        "observation/wrist_image": torch.randint(0, 256, (batch_size, img_size, img_size, 3), dtype=torch.uint8, device=device),
        # Loss masks
        "loss_mask": torch.ones(batch_size, 1, dtype=torch.bool, device=device),
        "loss_mask_sum": torch.ones(batch_size, 1, device=device) * batch_size,
        # Advantages and returns
        "advantages": torch.randn(batch_size, 1, device=device),
        "returns": torch.randn(batch_size, 1, device=device),
    }
    
    # awr_forward requires actions tensor (model-space actions for flow-matching)
    if forward_type == "awr_forward":
        data["actions"] = torch.randn(batch_size, action_horizon, action_dim_full, device=device)
    
    return data


def setup_distributed():
    """Setup distributed environment."""
    if not dist.is_initialized():
        # Set default environment variables if not set
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29500"

        dist.init_process_group(backend="nccl")

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


@hydra.main(
    version_base="1.1",
    config_path=".",
    config_name="test_ddp_openpi",
)
def main(cfg) -> None:
    """Main test function."""
    rank, local_rank, world_size = setup_distributed()

    try:
        # Skip full validate_cfg as it requires env/rollout setup
        # Just use the config directly for forward/backward test
        forward_type = cfg.algorithm.get("forward_type", "default_forward")
        
        if rank == 0:
            print("=" * 60)
            print(f"DDP OpenPi Forward/Backward Test ({forward_type})")
            print("=" * 60)
            print(f"Config:\n{json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2)[:2000]}...")
            print("=" * 60)

        # Import DDPModelManager
        from rlinf.workers.actor.ddp_actor_worker import DDPModelManager
        from rlinf.models import get_model

        # Create model manager
        if rank == 0:
            print(f"\n[Step 1] Creating DDPModelManager (rank={rank}, world_size={world_size})")

        manager = DDPModelManager(cfg.actor, world_size, rank)

        # Override model_provider_func to use the OpenPi model
        def model_provider_func():
            model = get_model(cfg.actor.model)
            if model is None:
                raise ValueError("Failed to load OpenPi model")
            return model

        manager.model_provider_func = model_provider_func

        # Setup model and optimizer
        if rank == 0:
            print("\n[Step 2] Setting up model and optimizer...")
        manager.setup_model_and_optimizer()

        if rank == 0:
            print(f"  - Model type: {type(manager.model)}")
            print(f"  - Model module type: {type(manager.model.module)}")
            num_params = sum(p.numel() for p in manager.model.parameters())
            num_trainable = sum(p.numel() for p in manager.model.parameters() if p.requires_grad)
            print(f"  - Total parameters: {num_params:,}")
            print(f"  - Trainable parameters: {num_trainable:,}")

        # Create mock data
        batch_size = cfg.actor.micro_batch_size
        device = torch.device(f"cuda:{local_rank}")

        if rank == 0:
            print(f"\n[Step 3] Creating mock data (batch_size={batch_size}, forward_type={forward_type})...")
        data = create_mock_data(batch_size, device, forward_type=forward_type)

        if rank == 0:
            print("  Mock data shapes:")
            for key, value in data.items():
                if torch.is_tensor(value):
                    print(f"    - {key}: {tuple(value.shape)}, dtype={value.dtype}")

        # Forward pass
        if rank == 0:
            print(f"\n[Step 4] Running forward pass ({forward_type})...")

        manager.model.train()
        manager.optimizer.zero_grad()

        with manager.amp_context:
            output_dict = manager.model(
                data=data,
                forward_type=forward_type,
                compute_logprobs=True,
                compute_entropy=cfg.algorithm.entropy_bonus > 0,
                compute_values=cfg.algorithm.adv_type == "gae",
                use_cache=False,
            )

        if rank == 0:
            print("  Forward pass output:")
            for key, value in output_dict.items():
                if torch.is_tensor(value):
                    print(f"    - {key}: {tuple(value.shape)}, dtype={value.dtype}")
                else:
                    print(f"    - {key}: {type(value)}")

        # Compute loss
        if rank == 0:
            print("\n[Step 5] Computing loss...")

        if forward_type == "awr_forward":
            # AWR-style loss: use ELBO directly as loss (MSE reconstruction loss)
            # The elbo from awr_forward is already per-element MSE loss
            elbo = output_dict["elbo"]
            loss = elbo.mean()
            metrics_data = {
                "elbo_mean": elbo.mean().item(),
                "elbo_std": elbo.std().item(),
            }
            if rank == 0:
                print(f"  - AWR Loss (ELBO mean): {loss.item():.6f}")
                print(f"  - Metrics: {metrics_data}")
        else:
            # Default forward: use policy_loss for PPO-style training
            loss_kwargs = {
                "loss_type": cfg.algorithm.loss_type,
                "logprob_type": cfg.algorithm.logprob_type,
                "reward_type": cfg.algorithm.reward_type,
                "single_action_dim": cfg.actor.model.get("action_dim", 7),
                "logprobs": output_dict["logprobs"],
                "values": output_dict.get("values", None),
                "old_logprobs": data["prev_logprobs"],
                "advantages": data["advantages"],
                "returns": data["returns"],
                "prev_values": data.get("prev_values", None),
                "clip_ratio_high": cfg.algorithm.clip_ratio_high,
                "clip_ratio_low": cfg.algorithm.clip_ratio_low,
                "value_clip": cfg.algorithm.get("value_clip", None),
                "huber_delta": cfg.algorithm.get("huber_delta", None),
                "loss_mask": data.get("loss_mask", None),
                "loss_mask_sum": data.get("loss_mask_sum", None),
                "max_episode_steps": cfg.env.train.get("max_episode_steps", 10),
                "task_type": cfg.runner.task_type,
                "critic_warmup": False,
            }

            loss, metrics_data = policy_loss(**loss_kwargs)

            if rank == 0:
                print(f"  - Policy Loss: {loss.item():.6f}")
                print(f"  - Metrics: {metrics_data}")

        # Backward pass
        if rank == 0:
            print("\n[Step 6] Running backward pass...")

        manager.grad_scaler.scale(loss).backward()

        # Check gradients
        grad_norms = []
        for name, param in manager.model.named_parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        if rank == 0:
            print(f"  - Number of parameters with gradients: {len(grad_norms)}")
            if grad_norms:
                print(f"  - Gradient norm (mean): {sum(grad_norms) / len(grad_norms):.6f}")
                print(f"  - Gradient norm (max): {max(grad_norms):.6f}")

        # Optimizer step
        if rank == 0:
            print("\n[Step 7] Running optimizer step...")

        grad_norm, lr_list = manager.optimizer_step()

        if rank == 0:
            print(f"  - Grad norm after clipping: {grad_norm:.6f}")
            print(f"  - Learning rates: {lr_list}")

        # Success
        if rank == 0:
            print("\n" + "=" * 60)
            print(f"✓ TEST PASSED: DDP {forward_type} with OpenPi successful!")
            print("=" * 60)

        dist.barrier()

    except Exception as e:
        print(f"\n✗ TEST FAILED on rank {rank}: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()

