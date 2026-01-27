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

import os
from contextlib import nullcontext
from typing import ContextManager, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import get_iterator_k_split
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    put_tensor_device,
)
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import (
    clear_memory,
    masked_mean,
    reshape_entropy,
    warmup_optimizer_state,
)


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


class DDPModelManager:
    """
    DDP Model Manager for RL training.
    Uses DistributedDataParallel without model sharding.
    """

    def __init__(self, cfg: DictConfig, world_size: int, rank: int) -> None:
        """
        Initialize DDP Model Manager.

        Args:
            cfg: actor config in yaml file.
            world_size: total number of DDP actor processes.
            rank: the rank of this process.
        """
        self._cfg = cfg
        self._logger = get_logger()
        self._world_size = world_size
        self._rank = rank
        self.torch_dtype = torch_dtype_from_precision(self._cfg.model.precision)

        self.optimizer_steps = 0
        self.critic_warmup_steps = 0
        if self._cfg.get("optim", {}).get(
            "critic_warmup_steps", None
        ) and self._cfg.model.get("add_value_head", False):
            self.critic_warmup_steps = self._cfg.optim.critic_warmup_steps
        self.store_requires_grad_param_name = []

        self.amp_context = self._create_amp_context()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        # Initialize distributed process group if not already initialized
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            self._logger.info(
                f"[DDP] Initialized process group: rank={rank}, world_size={world_size}"
            )

        self.is_weight_offloaded = False
        self.is_optimizer_offloaded = False

    def _create_amp_context(self) -> ContextManager:
        """
        Create AMP context manager based on configuration.

        Returns:
            A context manager for automatic mixed precision (AMP) if enabled,
            otherwise a null context manager.
        """
        amp_cfg = self._cfg.get("amp", {})
        if not amp_cfg.get("enabled", False):
            self._logger.info("[DDP] AMP is disabled.")
            return nullcontext()

        precision = torch_dtype_from_precision(amp_cfg.get("precision", "bf16"))
        self._logger.info(f"[DDP] AMP is enabled with precision: {precision}.")

        return torch.amp.autocast(device_type="cuda", dtype=precision)

    def model_provider_func(self) -> torch.nn.Module:
        """
        Initialize model used by DDP actor.

        Returns:
            model: the initialized model.
        """
        cfg = self._cfg

        assert torch.cuda.is_available(), "CUDA is not available."
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        model_config = AutoConfig.from_pretrained(
            cfg.model.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
            auto_model_class = AutoModelForVision2Seq
        else:
            auto_model_class = AutoModelForCausalLM

        model = auto_model_class.from_pretrained(
            cfg.model.model_path,
            torch_dtype=self.torch_dtype,
            config=model_config,
            trust_remote_code=True,
        )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        return model

    def setup_model_and_optimizer(self) -> None:
        """
        Setup model, lr_scheduler, optimizer and grad_scaler.
        """
        module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self._cfg.get("gradient_checkpointing", False):
            self._logger.info("[DDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
        else:
            self._logger.info("[DDP] Gradient checkpointing is disabled")

        # Move model to device
        module = module.to(self.device)

        # Wrap model with DDP
        self.model = DDP(
            module,
            device_ids=[self.device],
            output_device=self.device,
            find_unused_parameters=self._cfg.get("find_unused_parameters", False),
            broadcast_buffers=self._cfg.get("broadcast_buffers", True),
        )

        self.optimizer = self.build_optimizer(
            model=self.model, enable_critic_warmup=self.critic_warmup_steps > 0
        )
        self.lr_scheduler = self.build_lr_scheduler(optimizer=self.optimizer)

        amp_cfg = self._cfg.get("amp", {})
        self.grad_scaler = self.build_grad_scaler(amp_cfg.get("use_grad_scaler", False))

    def get_model_state_dict(self, cpu_offload: bool, full_state_dict: bool) -> dict:
        """
        Get the model state dict.

        Args:
            cpu_offload: Whether to offload the state dict to CPU.
            full_state_dict: Whether to get the full state dict (ignored for DDP,
                always returns full state dict).

        Returns:
            dict: The state dict of the model.
        """
        state_dict = self.model.module.state_dict()
        if cpu_offload:
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
        return state_dict

    def load_checkpoint(self, load_path: str) -> None:
        """
        Load checkpoint from local path.

        Args:
            load_path: the path to load checkpoint.
        """
        checkpoint = torch.load(load_path, map_location=f"cuda:{self.device}")
        self.model.module.load_state_dict(checkpoint.get("model", checkpoint))
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """
        Save checkpoint to local path.
        Only rank 0 saves the checkpoint.

        Args:
            save_path: the directory to save checkpoint.
            step: the current training step.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        if torch.distributed.get_rank() == 0:
            os.makedirs(save_path, exist_ok=True)
            checkpoint = {
                "model": self.model.module.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "step": step,
            }
            torch.save(checkpoint, os.path.join(save_path, f"checkpoint_{step}.pt"))

        torch.distributed.barrier()

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        """
        Offload model parameters and gradients to CPU.

        Args:
            offload_grad: whether to offload gradients.
        """
        for param in self.model.parameters():
            param.data = param.data.cpu()
            if offload_grad and param.grad is not None:
                param.grad = param.grad.cpu()
        self.is_weight_offloaded = True

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        """
        Load parameters and gradients to the specified device.

        Args:
            device_id: the target device id.
            load_grad: whether to load gradients.
        """
        device = torch.device(f"cuda:{device_id}")
        for param in self.model.parameters():
            param.data = param.data.to(device)
            if load_grad and param.grad is not None:
                param.grad = param.grad.to(device)
        self.is_weight_offloaded = False

    def offload_optimizer(self) -> None:
        """
        Offload optimizer states to CPU.
        """
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()
        self.is_optimizer_offloaded = True

    def load_optimizer(self, device_id: int) -> None:
        """
        Load optimizer states to the specified device.

        Args:
            device_id: the target device id.
        """
        device = torch.device(f"cuda:{device_id}")
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        self.is_optimizer_offloaded = False

    def optimizer_step(self) -> tuple[float, list[float]]:
        """
        Perform optimizer step.

        Returns:
            A tuple of (grad_norm, lr_list).
        """
        self.optimizer_steps += 1
        self.grad_scaler.unscale_(optimizer=self.optimizer)

        # Clip gradients
        max_grad_norm = self._cfg.optim.get("max_grad_norm", 1.0)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_grad_norm
        )

        if not torch.isfinite(grad_norm):
            self._logger.warning(
                f"[DDP] Non-finite grad norm {grad_norm} detected. Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)

        self.grad_scaler.update()

        if self.critic_warmup_steps > 0:
            lr_list = [0.0 for _ in self.optimizer.param_groups]
            if self.optimizer_steps >= self.critic_warmup_steps:
                self.optimizer = self.build_optimizer(model=self.model)
                self.critic_warmup_steps = 0
        else:
            lr_list = [group["lr"] for group in self.optimizer.param_groups]

        # Convert grad_norm to float for numpy compatibility
        if torch.is_tensor(grad_norm):
            grad_norm = grad_norm.item()

        return grad_norm, lr_list

    def build_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        """
        Build the learning rate scheduler.

        Args:
            optimizer: The optimizer for which to schedule the learning rate.

        Returns:
            LRScheduler: The learning rate scheduler.
        """
        total_steps = self._cfg.optim.get("total_training_steps", 0)
        num_warmup_steps = int(self._cfg.optim.get("lr_warmup_steps", -1))
        lr_scheduler = self._cfg.optim.get("lr_scheduler", "constant")
        num_cycles = self._cfg.optim.get("num_cycles", 0.5)
        min_lr = self._cfg.optim.get("min_lr", 0.0)
        min_lr_rate = self._cfg.optim.get("min_lr_rate", None)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = self._cfg.optim.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        return get_lr_scheduler(
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
            min_lr=min_lr,
            min_lr_rate=min_lr_rate,
        )

    def build_optimizer(
        self,
        model: Union[nn.Module, DDP],
        enable_critic_warmup: bool = False,
    ) -> Optimizer:
        """
        Build the optimizer.

        Args:
            model: The model to optimize.
            enable_critic_warmup: Whether to enable critic warmup.

        Returns:
            Optimizer: The constructed optimizer.
        """
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        adam_eps = self._cfg.optim.get("adam_eps", 1e-8)
        weight_decay = self._cfg.optim.get("weight_decay", 1e-2)

        params_actor = []
        params_critic = []

        if enable_critic_warmup:
            self._logger.info("[DDP] Enable critic warmup for value head.")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.store_requires_grad_param_name.append(name)
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                        continue
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if name in self.store_requires_grad_param_name:
                    param.requires_grad = True
                if param.requires_grad:
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)

        param_groups = []
        if len(params_actor) > 0:
            param_groups.append(
                {
                    "params": params_actor,
                    "lr": self._cfg.optim.lr,
                    "betas": betas,
                }
            )
        if len(params_critic) > 0:
            param_groups.append(
                {
                    "params": params_critic,
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            )
        optimizer = torch.optim.AdamW(
            param_groups,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        warmup_optimizer_state(optimizer)
        return optimizer

    def build_grad_scaler(self, enabled: bool) -> GradScaler:
        """
        Build the gradient scaler.

        Args:
            enabled: Whether to enable gradient scaling.

        Returns:
            GradScaler: The gradient scaler.
        """
        return GradScaler(enabled=enabled)

    def before_micro_batch(
        self, model: DDP, is_last_micro_batch: bool
    ) -> ContextManager:
        """
        Setup context manager before processing a micro-batch.
        Uses no_sync() for non-last micro-batches to avoid gradient synchronization.

        Args:
            model: The DDP model.
            is_last_micro_batch: Whether this is the last micro-batch.

        Returns:
            A context manager for the micro-batch processing.
        """
        if is_last_micro_batch:
            return nullcontext()
        else:
            return model.no_sync()


class EmbodiedDDPActor(DDPModelManager, Worker):
    """
    Embodied actor worker using DDP (no model sharding).
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num

        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """
        Setup destination ranks for weight communication.
        """
        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def init_worker(self) -> None:
        """
        Initialize the actor worker.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def sync_model_to_rollout(self) -> None:
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    def recv_rollout_batch(self, input_channel: Channel) -> None:
        """
        Receive rollout batch from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        send_num = self._component_placement.get_world_size("rollout") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        self.rollout_batch = {}
        recv_list = []
        for _ in range(split_num):
            recv_list.append(input_channel.get())

        # shape [num_chunk, bsz, chunk_size], cat dim 1
        self.rollout_batch = cat_list_of_dict_tensor(recv_list, dim=1)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.algorithm.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        # Split to make minibatch iterator for updating the actor
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = get_iterator_k_split(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = get_iterator_k_split(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, data in enumerate(train_micro_batch):
                    data = put_tensor_device(
                        data, f"cuda:{int(os.environ['LOCAL_RANK'])}"
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )
                    advantages = data["advantages"]
                    prev_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_sum = data.get("loss_mask_sum", None)

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        data["temperature"] = (
                            self.cfg.algorithm.sampling_params.temperature_train
                        )
                        data["top_k"] = self.cfg.algorithm.sampling_params.top_k

                    compute_values = (
                        True if "gae" in self.cfg.algorithm.adv_type else False
                    )

                    with self.amp_context:
                        output_dict = self.model(
                            data=data,
                            forward_type=self.cfg.algorithm.get(
                                "forward_type", "default_forward"
                            ),
                            compute_logprobs=True,
                            compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                            compute_values=compute_values,
                            use_cache=False,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T
                    ]:
                        prev_logprobs = output_dict["prev_logprobs"]

                    kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": output_dict["logprobs"],
                        "values": output_dict.get("values", None),
                        "old_logprobs": prev_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": prev_values,
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_sum": loss_mask_sum,
                        "max_episode_steps": self.cfg.env.train.max_episode_steps,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }
                    loss, metrics_data = policy_loss(**kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not kwargs["critic_warmup"]
                    ):
                        entropy = output_dict["entropy"]
                        entropy = reshape_entropy(
                            entropy,
                            entropy_type=self.cfg.algorithm.entropy_type,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=output_dict["logprobs"].shape[0],
                        )
                        entropy_loss = masked_mean(entropy, mask=loss_mask)
                        loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
                    metrics_data["entropy_loss"] = entropy_loss.detach().item()

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def set_global_step(self, global_step) -> None:
        """
        Set the global step for the model, if needed.
        """
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)
        elif hasattr(self.model.module, "set_global_step"):
            self.model.module.set_global_step(global_step)

