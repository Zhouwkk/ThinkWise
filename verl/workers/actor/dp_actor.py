# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        return log_probs

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs

    @torch.no_grad()
    def compute_mar(self, data: DataProto) -> list:
        """Compute MAR by piggybacking on a flash-attention forward pass.

        Supports two modes (controlled by self.config.mar_mode):
          - 'mid': MAR_mid — average over middle 50% layers, uniform temporal mean
          - 'vsh': MAR_VSH — only top-K visual-sensitive heads, topk temporal aggregation
        """
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb

        self.actor_module.eval()

        VISION_TOKEN_ID = 151655
        T = 64

        mar_mode = getattr(self.config, "mar_mode", "mid")
        vsh_heads = None
        if mar_mode == "vsh":
            vsh_heads = [(l, h) for l, h in getattr(self.config, "mar_vsh_heads",
                         [[33,13],[32,5],[12,3],[34,11],[33,9],[27,3],[35,1],[34,4]])]

        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]
        data = data.select(select_keys, non_tensor_select_keys)

        language_model = self.actor_module.model.language_model
        num_layers = len(language_model.layers)

        # Determine which layers to hook
        if mar_mode == "vsh":
            hook_layers = sorted(set(l for l, h in vsh_heads))
        elif mar_mode == "full_topk":
            hook_layers = list(range(num_layers))
        else:  # mid
            mid_start = num_layers // 4
            mid_end = 3 * num_layers // 4
            hook_layers = list(range(mid_start, mid_end))
        num_hook_layers = len(hook_layers)

        micro_batches = data.split(1)
        mar_values = []

        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute MAR", position=1)

        for micro_batch in micro_batches:
            input_ids = micro_batch.batch["input_ids"]
            attention_mask = micro_batch.batch["attention_mask"]
            position_ids = micro_batch.batch["position_ids"]
            responses = micro_batch.batch["responses"]
            response_length = responses.size(-1)
            seqlen = input_ids.size(1)

            vision_mask = (input_ids[0] == VISION_TOKEN_ID)
            if not vision_mask.any():
                mar_values.append(0.0)
                continue

            T_actual = min(T, response_length)
            prompt_len = seqlen - response_length
            truncated_len = prompt_len + T_actual

            input_ids_t = input_ids[:, :truncated_len]
            attention_mask_t = attention_mask[:, :truncated_len]
            if position_ids.dim() == 3:
                position_ids_t = position_ids[:, :, :truncated_len].transpose(0, 1)
            else:
                position_ids_t = position_ids[:, :truncated_len]

            vision_mask_t = (input_ids_t[0] == VISION_TOKEN_ID)
            vision_positions = vision_mask_t.nonzero(as_tuple=True)[0]
            resp_positions = torch.arange(prompt_len, truncated_len, device=input_ids.device)

            multi_modal_inputs = {}
            if "multi_modal_inputs" in micro_batch.non_tensor_batch:
                for key in micro_batch.non_tensor_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inp[key] for inp in micro_batch.non_tensor_batch["multi_modal_inputs"]], dim=0
                    )

            try:
                # ── Step 1: Register hooks to capture raw Q/K from q_proj/k_proj ──
                captured_qk = {}
                hooks = []

                for layer_idx in hook_layers:
                    captured_qk[layer_idx] = {}
                    attn = language_model.layers[layer_idx].self_attn

                    def make_q_hook(lid):
                        def hook_fn(module, input, output):
                            captured_qk[lid]["q"] = output.detach()
                        return hook_fn

                    def make_k_hook(lid):
                        def hook_fn(module, input, output):
                            captured_qk[lid]["k"] = output.detach()
                        return hook_fn

                    hooks.append(attn.q_proj.register_forward_hook(make_q_hook(layer_idx)))
                    hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))

                # Capture RoPE cos/sin
                captured_rope = {}
                def rope_hook(module, args, output):
                    captured_rope["cos"] = output[0].detach()
                    captured_rope["sin"] = output[1].detach()
                hooks.append(language_model.rotary_emb.register_forward_hook(rope_hook))

                # ── Step 2: Flash-attention forward pass (same as compute_log_prob) ──
                if self.config.padding_free:
                    from flash_attn.bert_padding import unpad_input, index_first_axis
                    from einops import rearrange

                    input_ids_rmpad, indices, *_ = unpad_input(
                        input_ids_t.unsqueeze(-1), attention_mask_t
                    )
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)

                    if position_ids_t.dim() == 3:
                        position_ids_rmpad = (
                            index_first_axis(
                                rearrange(position_ids_t, "c b s ... -> (b s) c ..."), indices
                            ).transpose(0, 1).unsqueeze(1)
                        )
                    else:
                        position_ids_rmpad = index_first_axis(
                            rearrange(position_ids_t.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                        ).transpose(0, 1)

                    try:
                        self.actor_module(
                            input_ids=input_ids_rmpad,
                            attention_mask=None,
                            position_ids=position_ids_rmpad,
                            **multi_modal_inputs,
                            use_cache=False,
                        )
                    finally:
                        for h in hooks:
                            h.remove()

                    # Map positions from padded to unpadded
                    pad_to_unpad = torch.full((truncated_len,), -1, device=input_ids.device, dtype=torch.long)
                    pad_to_unpad[indices] = torch.arange(len(indices), device=input_ids.device)
                    vis_pos = pad_to_unpad[vision_positions]
                    resp_pos = pad_to_unpad[resp_positions]
                    vis_pos = vis_pos[vis_pos >= 0]
                    resp_pos = resp_pos[resp_pos >= 0]
                else:
                    try:
                        self.actor_module(
                            input_ids=input_ids_t,
                            attention_mask=attention_mask_t,
                            position_ids=position_ids_t,
                            **multi_modal_inputs,
                            use_cache=False,
                        )
                    finally:
                        for h in hooks:
                            h.remove()

                    vis_pos = vision_positions
                    resp_pos = resp_positions

                # ── Step 3: Compute MAR from captured Q/K (no model weights needed) ──
                if not captured_qk or not captured_rope:
                    mar_values.append(0.0)
                    continue

                cos, sin = captured_rope["cos"], captured_rope["sin"]
                attn_cfg = language_model.layers[hook_layers[0]].self_attn
                head_dim = attn_cfg.head_dim
                n_heads = attn_cfg.num_heads
                n_kv_heads = attn_cfg.num_key_value_heads
                n_kv_groups = attn_cfg.num_key_value_groups
                scaling = attn_cfg.scaling
                mrope_section = attn_cfg.rope_scaling["mrope_section"]

                if mar_mode == "vsh":
                    # MAR_VSH: collect per-head signals for VSH heads, then topk temporal aggregation
                    head_signals = []  # list of (T_actual,) tensors, one per VSH head
                    for layer_idx in hook_layers:
                        qk = captured_qk.pop(layer_idx, None)
                        if qk is None or "q" not in qk or "k" not in qk:
                            continue
                        raw_q = qk["q"]
                        raw_k = qk["k"]
                        q = raw_q.view(1, -1, n_heads, head_dim).transpose(1, 2)
                        k = raw_k.view(1, -1, n_kv_heads, head_dim).transpose(1, 2)
                        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
                        if n_kv_groups > 1:
                            k = k.repeat_interleave(n_kv_groups, dim=1)
                        q_resp = q[:, :, resp_pos, :]
                        attn_full = torch.matmul(q_resp, k.transpose(-2, -1)) * scaling
                        attn_probs = torch.softmax(attn_full.float(), dim=-1)
                        attn_to_vis = attn_probs[:, :, :, vis_pos]  # (1, H, T, V)
                        # sum over vision tokens → (1, H, T)
                        attn_sum_vis = attn_to_vis.sum(dim=-1)
                        for l, h in vsh_heads:
                            if l == layer_idx:
                                head_signals.append(attn_sum_vis[0, h, :].cpu())
                        del raw_q, raw_k, q, k, q_resp, attn_full, attn_probs, attn_to_vis, attn_sum_vis

                    if head_signals and T_actual > 0:
                        signals = torch.stack(head_signals, dim=0)  # (K, T_actual)
                        k_topk = max(1, T_actual // 4)
                        token_mean = signals.mean(dim=0)
                        topk_idx = torch.argsort(token_mean)[-k_topk:]
                        mar_val = float(signals[:, topk_idx].mean().item())
                    else:
                        mar_val = 0.0

                else:
                    # MAR_full+topk: all layers, topk temporal aggregation
                    # MAR_mid: middle 50% layers, uniform temporal mean
                    mar_layer_signals = []  # (T_actual,) per layer, mean over heads
                    mar_sum = 0.0
                    for layer_idx in hook_layers:
                        qk = captured_qk.pop(layer_idx, None)
                        if qk is None or "q" not in qk or "k" not in qk:
                            continue
                        raw_q = qk["q"]
                        raw_k = qk["k"]
                        q = raw_q.view(1, -1, n_heads, head_dim).transpose(1, 2)
                        k = raw_k.view(1, -1, n_kv_heads, head_dim).transpose(1, 2)
                        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section)
                        if n_kv_groups > 1:
                            k = k.repeat_interleave(n_kv_groups, dim=1)
                        q_resp = q[:, :, resp_pos, :]
                        attn_full = torch.matmul(q_resp, k.transpose(-2, -1)) * scaling
                        attn_probs = torch.softmax(attn_full.float(), dim=-1)
                        attn_to_vis = attn_probs[:, :, :, vis_pos]
                        if mar_mode == "full_topk":
                            # sum over vision tokens, mean over heads → (T_actual,)
                            mar_layer_signals.append(attn_to_vis.sum(dim=-1).mean(dim=1)[0].cpu())
                        else:
                            mar_sum += attn_to_vis.sum().item()
                        del raw_q, raw_k, q, k, q_resp, attn_full, attn_probs, attn_to_vis

                    if mar_mode == "full_topk":
                        if mar_layer_signals and T_actual > 0:
                            token_mean = torch.stack(mar_layer_signals, dim=0).mean(dim=0)  # (T_actual,)
                            k_topk = max(1, T_actual // 4)
                            topk_idx = torch.argsort(token_mean)[-k_topk:]
                            mar_val = float(token_mean[topk_idx].mean().item())
                        else:
                            mar_val = 0.0
                    else:
                        n_vision = len(vis_pos)
                        if n_vision > 0 and T_actual > 0 and num_hook_layers > 0 and n_heads > 0:
                            mar_val = mar_sum / (num_hook_layers * n_heads * T_actual)
                        else:
                            mar_val = 0.0

                del captured_rope
                mar_values.append(mar_val)

                if len(mar_values) <= 3 and self.rank == 0:
                    print(f"[MAR Debug] sample {len(mar_values)}: mar={mar_val:.6f} (mode={mar_mode}), "
                          f"T_actual={T_actual}, n_vision={len(vis_pos)}, n_hook_layers={num_hook_layers}")

            except Exception as e:
                if len(mar_values) < 3 and self.rank == 0:
                    print(f"[MAR] Warning: failed for sample {len(mar_values)+1}, error: {e}")
                    import traceback
                    traceback.print_exc()
                mar_values.append(0.0)

        nonzero = [v for v in mar_values if v > 0]
        if mar_values and self.rank == 0:
            mean_val = sum(mar_values) / len(mar_values)
            print(f"[MAR Summary] mode={mar_mode}, total={len(mar_values)}, nonzero={len(nonzero)}, "
                  f"mean={mean_val:.6f}, max={max(mar_values):.6f}")

        torch.cuda.empty_cache()
        return mar_values

    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        select_keys.extend(["old_log_probs", "ref_log_probs", "advantages"])
        non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                if self.config.dynamic_batching:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_probs = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    # all return: (bsz, response_length)
                    log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        
                        # Use dynamic KL coef if available, otherwise use config value
                        if "dynamic_kl_coef" in model_inputs:
                            # dynamic_kl_coef is a list (could be same or different values)
                            kl_coef_list = model_inputs["dynamic_kl_coef"]
                            if isinstance(kl_coef_list, list) and len(kl_coef_list) > 0:
                                # For micro-batch, we need to get the right subset of kl_coefs
                                # Get the indices for current micro batch
                                batch_start_idx = model_inputs.get("_batch_start_idx", 0)
                                micro_batch_size = response_mask.shape[0]
                                
                                # Get KL coefs for this micro batch
                                micro_batch_kl_coefs = kl_coef_list[batch_start_idx:batch_start_idx + micro_batch_size]
                                
                                # Check if all values are the same (batch-level) or different (group-level)
                                unique_coefs = list(set(micro_batch_kl_coefs))
                                if len(unique_coefs) == 1:
                                    # All same - batch-level KL
                                    kl_coef = unique_coefs[0]
                                    print(f"[Actor] Using batch-level dynamic KL coef: {kl_coef:.6f}")
                                else:
                                    # Different values - group-level KL
                                    # For now, use average for the loss computation
                                    # TODO: Could implement per-token KL penalty if needed
                                    kl_coef = sum(micro_batch_kl_coefs) / len(micro_batch_kl_coefs)
                                    print(f"[Actor] Using group-level dynamic KL coefs:")
                                    print(f"  Micro batch size: {micro_batch_size}")
                                    print(f"  Unique KL coefs in micro batch: {len(unique_coefs)}")
                                    print(f"  KL coef range: [{min(micro_batch_kl_coefs):.6f}, {max(micro_batch_kl_coefs):.6f}]")
                                    print(f"  Average KL coef for loss: {kl_coef:.6f}")
                            else:
                                kl_coef = self.config.kl_coef
                                print(f"[Actor] WARNING: dynamic_kl_coef is empty or not a list, using config value")
                        else:
                            kl_coef = self.config.kl_coef
                            
                        loss = pg_loss + kl_loss * kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
    
    def update_kl_coef(self, new_kl_coef: float):
        """Update the KL coefficient in the config.
        
        Args:
            new_kl_coef: The new KL coefficient value
        """
        self.config.kl_coef = new_kl_coef
        print(f"[Actor] KL coef updated to: {new_kl_coef:.6f}")
