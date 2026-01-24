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

import logging
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import numpy as np
import torch
import torch.nn.functional as F
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead

logger = logging.getLogger(__name__)

def _shape_info(value: Any) -> Any:
    if torch.is_tensor(value):
        return tuple(value.shape)
    if isinstance(value, np.ndarray):
        return value.shape
    if isinstance(value, (list, tuple)):
        return [_shape_info(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape_info(val) for key, val in value.items()}
    if hasattr(value, "shape"):
        try:
            return tuple(value.shape)
        except Exception:
            return type(value).__name__
    return "scalar"

def _dtype_info(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.dtype
    if isinstance(value, np.ndarray):
        return value.dtype
    if isinstance(value, (list, tuple)):
        return [_dtype_info(item) for item in value]
    if isinstance(value, dict):
        return {key: _dtype_info(val) for key, val in value.items()}
    if hasattr(value, "dtype"):
        return value.dtype
    return "scalar"

def _device_info(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.device
    if isinstance(value, np.ndarray):
        return "not a tensor"
    if isinstance(value, (list, tuple)):
        return [f"device: {_device_info(item)}" for item in value]
    if isinstance(value, dict):
        return {key: f"device: {_device_info(val)}" for key, val in value.items()}
    return "not a tensor"

@dataclass(frozen=True)
class OpenPi0Config(Pi0Config):
    # config for rl
    config_name: str = "pi0_libero"  # pi0_libero, pi05_libero, pi0_maniskill, pi05_maniskill, pi0_metaworld, pi05_metaworld
    num_images_in_input: int = 2  # number of images in input
    noise_method: str = "flow_sde"  # flow_sde, flow_noise, flow_cps
    # noise config for flow-sde
    noise_level: float = 0.5
    noise_anneal: bool = False
    noise_params: list = field(
        default_factory=lambda: [0.7, 0.3, 400]
    )  # noise_start, noise_end, noise_anneal_steps
    # noise config for flow-noise
    noise_logvar_range: list = field(
        default_factory=lambda: [0.08, 0.16]
    )  # [min_std, max_std]
    # hyper-parameters
    action_chunk: int = 5  # action chunk
    action_env_dim: int = 7  # for environment action dim
    num_steps: int = 10  # denoise steps
    # training config
    train_expert_only: bool = False
    safe_get_logprob: bool = False
    joint_logprob: bool = False  # designed for flow-noise
    double_layer: bool = False  # designed for flow-sde without acceleration
    ignore_last: bool = False  # ignore the last action for noise injection
    # critic
    detach_critic_input: bool = False  # detach critic input with the action expert
    chunk_critic_input: bool = False  # use only the action chunk for critic estimation
    add_value_head: bool = False  # add value head for ppo
    value_after_vlm: bool = False  # value after vlm, pi05 mode
    value_vlm_mode: str = "mean_token"  # last_token, mean_token, first_token


class OpenPi0ForRLActionPrediction(BasePolicy, PI0Pytorch):
    """
    Pi0 model for reinforcement learning action prediction.
    """

    config: OpenPi0Config

    @property
    def _no_split_modules(self) -> list[str]:
        if self.config.train_expert_only:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        if self.config.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "action_in_proj",
            "action_out_proj",
            "lm_head",
            # --pi0 only--
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
            # --pi05 only--
            "time_mlp_in",
            "time_mlp_out",
        ]

    def __init__(
        self,
        config: OpenPi0Config,
    ):
        # Override `sample_actions` to prevent parent class polymorphic call
        sample_actions_func = self.sample_actions
        PI0Pytorch.__init__(self, config)
        self.sample_actions = sample_actions_func
        self.global_step = 0
        # assert
        assert not (self.config.double_layer and self.config.joint_logprob), (
            "double_layer and joint_logprob can not be set at the same time"
        )

        # rl model init
        if self.config.value_after_vlm:
            proj_width = 2048
        else:
            proj_width = 1024
        # value head
        if self.config.add_value_head:
            if self.config.config_name == "pi05_maniskill":
                value_head_hidden_sizes = (1024, 512, 256)
            else:
                value_head_hidden_sizes = (512, 256, 128)
            value_head_activation = "relu"
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation=value_head_activation,
                bias_last=True,
            )
            self.value_head = self.value_head.to(
                dtype=self.action_out_proj.weight.dtype
            )
        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        # noise head for flow-noise
        if self.config.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=1024,
                out_dim=self.config.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=self.config.noise_logvar_range,
                noise_scheduler_type="learn",
            )
            self.noise_head = self.noise_head.to(
                dtype=self.action_out_proj.weight.dtype
            )

        for name, module in self.named_modules():
            # Set _fsdp_wrap_name to the last part of the path (e.g., "model.action_in_proj" -> "action_in_proj")
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def _tensor_to_numpy(self, x):
        """Convert tensor to numpy, handling BFloat16/Float16 conversion."""
        if torch.is_tensor(x):
            x_cpu = x.detach().cpu()
            # BFloat16 and Float16 are not supported by numpy, convert to float32
            if x_cpu.dtype in (torch.bfloat16, torch.float16):
                x_cpu = x_cpu.float()
            return np.asarray(x_cpu)
        return x

    def _tensor_to_numpy_single(self, x, index):
        """Convert single tensor element to numpy, handling BFloat16/Float16 conversion."""
        if torch.is_tensor(x):
            x_cpu = x[index].detach().cpu()
            # BFloat16 and Float16 are not supported by numpy, convert to float32
            if x_cpu.dtype in (torch.bfloat16, torch.float16):
                x_cpu = x_cpu.float()
            return np.asarray(x_cpu)
        return x[index]

    def setup_wrappers(
        self,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
    ):
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

    def input_transform(self, obs: dict, transpose=True):
        inputs = jax.tree.map(lambda x: x, obs)
        # process input
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs.keys() if "/" in key}

        # tensor -> numpy (Convert BFloat16/Float16 to float32 for numpy compatibility)
        inputs = jax.tree.map(self._tensor_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))
        # split & transform
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: x[i], inputs)
            if transpose:
                # convert from [3,256,256] -> [256,256,3]
                sample = jax.tree.map(
                    lambda x: x.transpose(1, 2, 0)
                    if len(x.shape) == 3 and transpose
                    else x,
                    sample,
                )
            else:
                sample = jax.tree.map(lambda x: x if len(x.shape) == 3 else x, sample)
            if first_process:
                sample["prompt"] = obs["prompt"][i]
            else:
                sample["prompt"] = "xxxx"
            transformed_sample = self._input_transform(sample)
            transformed_samples.append(transformed_sample)
        # recombine
        inputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        # inputs = jax.tree.map(lambda *x: torch.stack(x, axis=0), inputs)
        if not first_process:
            inputs["tokenized_prompt"] = obs["tokenized_prompt"]
            inputs["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return inputs

    def output_transform(self, outputs):
        # split & transform
        batch_size = outputs["actions"].shape[0]
        transformed_samples = []
        for i in range(batch_size):
            sample = jax.tree.map(lambda x: self._tensor_to_numpy_single(x, i), outputs)
            sample = self._output_transform(sample)
            transformed_samples.append(sample)
        # recombine
        outputs = jax.tree.map(
            lambda *torch_arr: torch.from_numpy(np.asarray(torch_arr).copy()),
            *transformed_samples,
        )
        outputs["actions"] = outputs["actions"][:, : self.config.action_chunk]
        return outputs

    def forward(self, forward_type="default_forward", **kwargs):
        if forward_type == "sft_forward":
            return self.sft_forward(**kwargs)
        elif forward_type == "default_forward":
            return self.default_forward(**kwargs)
        elif forward_type == "awr_forward":
            return self.awr_forward_v3(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        observation = data["observation"]
        actions = data["actions"]
        return PI0Pytorch.forward(self, observation, actions)

    def awr_forward(self, data, **kwargs):
        observation = self.input_transform(data, transpose=False)
        # observation = self.precision_processor(observation)
        def _log_shape(name: str, value: Any) -> None:
            # pass
            logger.info("awr_forward %s shape: %s", name, _shape_info(value))
        
        def _check_nan(tensor: torch.Tensor, name: str) -> bool:
            """Check for NaN/Inf values and log detailed info if found."""
            if torch.is_tensor(tensor):
                has_nan = torch.isnan(tensor).any().item()
                has_inf = torch.isinf(tensor).any().item()
                if has_nan or has_inf:
                    nan_count = torch.isnan(tensor).sum().item()
                    inf_count = torch.isinf(tensor).sum().item()
                    total = tensor.numel()
                    logger.warning(
                        f"awr_forward NaN/Inf detected in {name}: "
                        f"NaN={nan_count}/{total} ({100*nan_count/total:.2f}%), "
                        f"Inf={inf_count}/{total} ({100*inf_count/total:.2f}%), "
                        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                        f"min={tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].min().item() if (total - nan_count - inf_count) > 0 else 'N/A'}, "
                        f"max={tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)].max().item() if (total - nan_count - inf_count) > 0 else 'N/A'}"
                    )
                    return True
            return False
        
        # _log_shape("data", data)
        # logger.info("awr_forward observation dtype before from_dict: %s", _dtype_info(observation))
        # logger.info("awr_forward observation devices before from_dict: %s", _device_info(observation))
        # _log_shape("observation before from_dict", observation)
        chains = data["chains"]
        device = chains.device
        observation = _model.Observation.from_dict(observation, device=device)  
        # TODO: data is sent back to CPU here due to the decorator in Observation class, need to fix it.
        
        
        actions = data["actions"].to(device).contiguous()
        _check_nan(actions, "actions")
        
        # logger.info("awr_forward observation images device after from_dict: %s", _device_info(observation.images))
        # logger.info("awr_forward observation image masks device after from_dict: %s", _device_info(observation.image_masks))
        # _log_shape("observation after from_dict", observation)
        # logger.info("awr_forward actions device: %s", _device_info(actions))


        # PI0Pytorch.forward returns per-step reconstruction loss (MSE) which is the
        # negative ELBO term for the flow-matching head. Use its negative as a
        # surrogate log-probability for AWR style weighting.
        # elbo = PI0Pytorch.forward(self, observation, actions)

        # copying PI0Pytorch.forward implementation but without requiring input being
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        ) 
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        _check_nan(state, "state")

        # if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
        _check_nan(noise, "noise")

        # if time is None:
        time = self.sample_time(actions.shape[0], actions.device)
        _check_nan(time, "time")

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        _check_nan(x_t, "x_t")
        _check_nan(u_t, "u_t")

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        _check_nan(prefix_embs, "prefix_embs")
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        _check_nan(suffix_embs, "suffix_embs")
        
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=True,
        )
        _check_nan(prefix_output, "prefix_output")

        # _log_shape("prefix_output", prefix_output)
        
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            time,
        )
        # _check_nan(suffix_out, "suffix_out")
        
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )  # [bs,n_action_steps,max_action_dim]
        _check_nan(v_t, "v_t")
        
        elbo = F.mse_loss(u_t, v_t, reduction="none")
        logger.info("elbo mean value: %s", v_t.mean().item())
        _check_nan(elbo, "elbo")
        # _log_shape("loss", loss)
        # value prediction
        # if (
        #     self.config.add_value_head
        #     and compute_values
        #     and not self.config.value_after_vlm
        # ):
        # raise ValueError("Stop here")

        # # Apply gradient checkpointing if enabled
        # def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        #     (_, suffix_out), _ = self.paligemma_with_expert.forward(
        #         attention_mask=att_2d_masks_4d,
        #         position_ids=position_ids,
        #         past_key_values=None,
        #         inputs_embeds=[prefix_embs, suffix_embs],
        #         use_cache=False,
        #         adarms_cond=[None, adarms_cond],
        #     )
        #     return suffix_out

        # suffix_out = self._apply_checkpoint(
        #     forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        # )

        # suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)

        # # Apply gradient checkpointing to final action projection if enabled
        # def action_out_proj_func(suffix_out):
        #     return self.action_out_proj(suffix_out)

        # v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # elbo = F.mse_loss(u_t, v_t, reduction="none")
        

        # Align shape with action chunk/env dim if present
        if elbo.dim() >= 3:
            logprobs = elbo[
                :, : self.config.action_chunk, : self.config.action_env_dim
            ]
        _check_nan(logprobs, "logprobs (final output)")
        _log_shape("logprobs", logprobs)
        _log_shape("elbo", elbo)
        _log_shape("data", data)

        return {
            "logprobs": logprobs,
            "values": None,
            "entropy": None,
            "elbo": elbo,
        }
        
    def awr_forward_v2(self, data, **kwargs):
        observation = self.input_transform(data, transpose=False)
        # observation = self.precision_processor(observation)
        # def _log_shape(name: str, value: Any) -> None:
        #     # pass
        #     logger.info("awr_forward %s shape: %s", name, _shape_info(value))
        
        # _log_shape("data", data)
        # logger.info("awr_forward observation dtype before from_dict: %s", _dtype_info(observation))
        # logger.info("awr_forward observation devices before from_dict: %s", _device_info(observation))
        # _log_shape("observation before from_dict", observation)
        chains = data["chains"]
        device = chains.device
        observation = _model.Observation.from_dict(observation, device=device)  
        # TODO: data is sent back to CPU here due to the decorator in Observation class, need to fix it.
        
        actions = data["actions"].to(device).contiguous()
        
        # logger.info("awr_forward observation images device after from_dict: %s", _device_info(observation.images))
        # logger.info("awr_forward observation image masks device after from_dict: %s", _device_info(observation.image_masks))
        # _log_shape("observation after from_dict", observation)
        # logger.info("awr_forward actions device: %s", _device_info(actions))


        # PI0Pytorch.forward returns per-step reconstruction loss (MSE) which is the
        # negative ELBO term for the flow-matching head. Use its negative as a
        # surrogate log-probability for AWR style weighting.
        # elbo = PI0Pytorch.forward(self, observation, actions)

        # copying PI0Pytorch.forward implementation but without requiring input being
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        ) 
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)

        # if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)

        # if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=True,
        )

        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            time,
        )
        # _check_nan(suffix_out, "suffix_out")
        
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )  # [bs,n_action_steps,max_action_dim]
        
        elbo = F.mse_loss(u_t, v_t, reduction="none")
        logger.info("elbo mean value: %s", v_t.mean().item())
        # _log_shape("loss", loss)
        # value prediction
        # if (
        #     self.config.add_value_head
        #     and compute_values
        #     and not self.config.value_after_vlm
        # ):
        # raise ValueError("Stop here")

        # # Apply gradient checkpointing if enabled
        # def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        #     (_, suffix_out), _ = self.paligemma_with_expert.forward(
        #         attention_mask=att_2d_masks_4d,
        #         position_ids=position_ids,
        #         past_key_values=None,
        #         inputs_embeds=[prefix_embs, suffix_embs],
        #         use_cache=False,
        #         adarms_cond=[None, adarms_cond],
        #     )
        #     return suffix_out

        # suffix_out = self._apply_checkpoint(
        #     forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        # )

        # suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)

        # # Apply gradient checkpointing to final action projection if enabled
        # def action_out_proj_func(suffix_out):
        #     return self.action_out_proj(suffix_out)

        # v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # elbo = F.mse_loss(u_t, v_t, reduction="none")
        

        # Align shape with action chunk/env dim if present
        if elbo.dim() >= 3:
            logprobs = elbo[
                :, : self.config.action_chunk, : self.config.action_env_dim
            ]

        return {
            "logprobs": logprobs,
            "values": None,
            "entropy": None,
            "elbo": elbo,
        }
        
    def awr_forward_v3(self, data, **kwargs):
        observation = self.input_transform(data, transpose=False)
        # observation = self.precision_processor(observation)
        # def _log_shape(name: str, value: Any) -> None:
        #     # pass
        #     logger.info("awr_forward %s shape: %s", name, _shape_info(value))
        
        # _log_shape("data", data)
        # logger.info("awr_forward observation dtype before from_dict: %s", _dtype_info(observation))
        # logger.info("awr_forward observation devices before from_dict: %s", _device_info(observation))
        # _log_shape("observation before from_dict", observation)
        chains = data["chains"]
        device = chains.device
        observation = _model.Observation.from_dict(observation, device=device)  
        # TODO: data is sent back to CPU here due to the decorator in Observation class, need to fix it.
        
        actions = data["actions"].to(device).contiguous()


        # PI0Pytorch.forward returns per-step reconstruction loss (MSE) which is the
        # negative ELBO term for the flow-matching head. Use its negative as a
        # surrogate log-probability for AWR style weighting.
        # elbo = PI0Pytorch.forward(self, observation, actions)

        # copying PI0Pytorch.forward implementation but without requiring input being
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        ) 
        device = next(self.parameters()).device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)

        # if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)

        # if time is None:
        time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
        
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)
        
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )  # [bs,n_action_steps,max_action_dim]
        
        elbo = F.mse_loss(u_t, v_t, reduction="none")
        logger.info("elbo mean value: %s", v_t.mean().item())
        # _log_shape("loss", loss)
        # value prediction
        # if (
        #     self.config.add_value_head
        #     and compute_values
        #     and not self.config.value_after_vlm
        # ):
        # raise ValueError("Stop here")

        # # Apply gradient checkpointing if enabled
        # def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
        #     (_, suffix_out), _ = self.paligemma_with_expert.forward(
        #         attention_mask=att_2d_masks_4d,
        #         position_ids=position_ids,
        #         past_key_values=None,
        #         inputs_embeds=[prefix_embs, suffix_embs],
        #         use_cache=False,
        #         adarms_cond=[None, adarms_cond],
        #     )
        #     return suffix_out

        # suffix_out = self._apply_checkpoint(
        #     forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        # )

        # suffix_out = suffix_out[:, -self.config.action_horizon :]
        # suffix_out = suffix_out.to(dtype=torch.float32)

        # # Apply gradient checkpointing to final action projection if enabled
        # def action_out_proj_func(suffix_out):
        #     return self.action_out_proj(suffix_out)

        # v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # elbo = F.mse_loss(u_t, v_t, reduction="none")
        

        # Align shape with action chunk/env dim if present
        if elbo.dim() >= 3:
            logprobs = elbo[
                :, : self.config.action_chunk, : self.config.action_env_dim
            ]

        return {
            "logprobs": logprobs,
            "values": None,
            "entropy": None,
            "elbo": elbo,
        }

    def default_forward(
        self,
        data: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:

        def _log_shape(name: str, value: Any) -> None:
            pass
            # logger.info("default_forward %s shape: %s", name, _shape_info(value))

        # get kwargs
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        _log_shape("compute_values", compute_values)
        _log_shape("chains", chains)
        _log_shape("denoise_inds", denoise_inds)
        _log_shape("data", data)
        # input transform
        observation = self.input_transform(data, transpose=False)
        _log_shape("observation_dict", observation)
        observation = _model.Observation.from_dict(observation)
        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )
        _log_shape("images", images)
        _log_shape("img_masks", img_masks)
        _log_shape("lang_tokens", lang_tokens)
        _log_shape("lang_masks", lang_masks)
        _log_shape("state", state)
        # transfer to device
        device = chains.device
        images = [img.to(device) for img in images]
        img_masks = [img_mask.to(device) for img_mask in img_masks]
        state = state.to(device)
        _log_shape("device", device)
        _log_shape("images_device", images)
        _log_shape("img_masks_device", img_masks)
        _log_shape("state_device", state)
        # get log prob
        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        _log_shape("log_probs", log_probs)
        _log_shape("value_t", value_t)
        _log_shape("entropy", entropy)
        log_probs = log_probs[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        entropy = entropy[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        _log_shape("log_probs_sliced", log_probs)
        _log_shape("entropy_sliced", entropy)
        # post process
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[
            :, None
        ]  # [:,None] to align with loss-mask shape
        value_t = value_t.mean(dim=-1, keepdim=False)
        _log_shape("log_probs_mean", log_probs)
        _log_shape("entropy_mean", entropy)
        _log_shape("value_t_mean", value_t)
        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def obs_processor(self, env_obs):
        # base observation
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        # state observation - ensure float32 to prevent BFloat16 conversion issues
        if "calvin" in self.config.config_name:
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
        else:
            state = env_obs["states"]
            if torch.is_tensor(state):
                state = state.to(dtype=torch.float32)
            processed_obs["observation/state"] = state
        # wrist image observation
        if env_obs["wrist_images"] is not None:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        # store used keys
        return processed_obs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        return_obs=True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        to_process_obs = self.obs_processor(env_obs)  # env obs -> policy input obs
        processed_obs = self.input_transform(
            to_process_obs, transpose=False
        )  # policy input obs -> model input obs
        processed_obs = self.precision_processor(
            processed_obs
        )  # obs precision processor
        observation = _model.Observation.from_dict(processed_obs)
        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform(
            {"actions": outputs["actions"], "state": observation.state}
        )["actions"].numpy()

        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
            "tokenized_prompt": processed_obs["tokenized_prompt"],
            "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"],
            "actions": outputs["actions"],  # model-space actions for AWR training
        }
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
        return actions, result

    @torch.no_grad()
    def sample_actions(
        self,
        observation: _model.Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> torch.Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        device = observation.state.device
        num_steps = self.config.num_steps
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_output, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        x_t = noise
        # add sde sample and traj collect
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        # add value based on the vlm for pi05, expert for pi0
        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output)
        if self.config.joint_logprob:
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        # In the joint logprob mode, we need to sample the logprob for each denoise step
        # In the non-joint logprob mode, only one denoise step is sampled and ode-sde mix sampling is used
        # denoise index
        if mode == "train":
            if self.config.joint_logprob:
                denoise_inds = torch.arange(num_steps)
            else:
                if self.config.ignore_last:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        # denoise step
        for idx in range(num_steps):
            # sample mean var val
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            # store
            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)
        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        # post process for logprob
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.config.action_chunk, : self.config.action_env_dim
        ]
        if self.config.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]
        # post process for value
        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """
        Sample the mean, variance and value of the action at a given timestep.
        Rollout sample (idx is int) and actor get_log_prob_value (idx is tensor) will load this function.
        """
        # expand the shape
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)
        # build parameters
        if self.config.noise_anneal:
            # noise annealing
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            # fixed noise level
            noise_level = torch.tensor(self.config.noise_level).to(device)
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        # input parameters
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]
        # velocity prediction
        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.action_out_proj(
            suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        )  # [bs,n_action_steps,max_action_dim]
        # value prediction
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            # use chunk critic input
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            # detach critic input
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        # ode sde mix sampling
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)
        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = (
            self.embed_suffix(state, x_t, timestep)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (
            "eager"  # noqa: SLF001
        )

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return suffix_out

    # TODO: to check potential nan here
    def get_logprob_norm(self, sample, mu, sigma):
        # logprob = log p(x|mu,sigma) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def preprocess_for_train(self, data):
        return data

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Compute image and language key value cache
        [prefix_output, _], past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        # get log prob
        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)
            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        # entropy is only available for flow-noise method
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)
        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output):
        # prefix_output:
        # pi05: [bs, (256 * 3 + 200) = 968, 2048]
        # pi0: [bs, (256 * 3 + 48) = 816, 1024]
        # token length
        if "pi05_" in self.config.config_name:
            lang_token_len = 200
            all_token_length = 968
        elif "pi0_" in self.config.config_name:
            lang_token_len = 48
            all_token_length = 816

        if self.config.value_vlm_mode == "mean_token":
            prefix_mask = (
                [True] * 256 * self.config.num_images_in_input
                + [False] * 256 * (3 - self.config.num_images_in_input)
                + [True] * lang_token_len
            )
        elif self.config.value_vlm_mode == "last_token":
            prefix_mask = [False] * (all_token_length - 1) + [True] * 1
        elif self.config.value_vlm_mode == "first_token":
            prefix_mask = [True] * 1 + [False] * (all_token_length - 1)
        prefix_out_value = prefix_output[:, prefix_mask, :]
        prefix_out_value = prefix_out_value.mean(dim=1, keepdim=False)
        prefix_out_value = prefix_out_value.to(dtype=torch.float32)
        values_vlm = self.value_head(prefix_out_value)[:, 0]
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def freeze_vlm(self):
        if self.config.train_expert_only:
            self.paligemma_with_expert.paligemma.eval()
            for params in self.paligemma_with_expert.paligemma.parameters():
                params.requires_grad = False
