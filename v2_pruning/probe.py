"""Read-only first-layer Q/K Probe for original LLaVA-1.5's Llama backbone.

Native reuse source: Hugging Face Transformers v4.37.2, commit
345b9b1a6a308a1fa6559251eb33ead2211240ac (modeling_llama.py blob
bd2269ce1606b6d1aff3b8771022703653318993):
https://github.com/huggingface/transformers/blob/345b9b1a6a308a1fa6559251eb33ead2211240ac/src/transformers/models/llama/modeling_llama.py

Borrow the model's existing input RMSNorm, Q/K projections and rotary module;
import apply_rotary_pos_emb/repeat_kv rather than reimplementing RoPE or GQA.
Contract: one full, unpruned, unpadded prefill sequence, no past KV cache,
original position IDs, native Llama with pretraining_tp=1 and no rope_scaling.
Production inputs/weights are FP16; FP32 is also accepted for small numerical
reference tests. This module creates no LM, V projection, MLP or decoder cache.
Only raw relevance is returned; near-constant signal fallback belongs to core.
"""

from __future__ import annotations

import math

import torch
import transformers
from torch import Tensor, nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)


class ProbeUnavailable(ValueError):
    """The caller must keep all original visual tokens for this sample."""


def _validate_indices(name: str, values: list[int], length: int) -> None:
    if not isinstance(values, list) or not values:
        raise ProbeUnavailable(f"{name} must be a nonempty list of sequence indices")
    if any(type(index) is not int or not 0 <= index < length for index in values):
        raise ProbeUnavailable(f"{name} contains a non-integer or out-of-range index")
    if len(set(values)) != len(values):
        raise ProbeUnavailable(f"{name} contains duplicate sequence indices")


def _require_finite(name: str, tensor: Tensor) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise ProbeUnavailable(f"{name} contains nonfinite values")


def _native_first_layer(model):
    if transformers.__version__ != "4.37.2":
        raise ProbeUnavailable("Probe requires transformers==4.37.2")
    if not callable(getattr(model, "get_model", None)):
        raise ProbeUnavailable("Probe requires the original LLaVA get_model() interface")
    try:
        layer = model.get_model().layers[0]
    except (AttributeError, IndexError, TypeError) as error:
        raise ProbeUnavailable("LLaVA first decoder layer is unavailable") from error
    if not isinstance(layer, LlamaDecoderLayer):
        raise ProbeUnavailable("Only the native Llama decoder layer is supported")
    attention = layer.self_attn
    if not isinstance(attention, LlamaAttention) or not isinstance(layer.input_layernorm, LlamaRMSNorm):
        raise ProbeUnavailable("Only native Llama input RMSNorm and attention are supported")
    if attention.config.model_type not in {"llama", "llava", "llava_llama"}:
        raise ProbeUnavailable("Unsupported model architecture; Qwen is not a Llama Probe target")
    if attention.config.pretraining_tp != 1:
        raise ProbeUnavailable("Probe only supports original pretraining_tp=1 Q/K projections")
    if attention.config.rope_scaling is not None or type(attention.rotary_emb) is not LlamaRotaryEmbedding:
        raise ProbeUnavailable("Probe only supports the original unscaled native Llama RoPE")
    if not isinstance(attention.q_proj, nn.Linear) or not isinstance(attention.k_proj, nn.Linear):
        raise ProbeUnavailable("Quantized or replacement Q/K projections are not supported")
    if layer.training or attention.training:
        raise ProbeUnavailable("The shared LLaVA model must already be in evaluation mode")
    return layer, attention


@torch.inference_mode()
def probe_relevance(
    model,
    inputs_embeds: Tensor,
    question_positions: list[int],
    visual_positions: list[int],
    position_ids: Tensor,
    query_chunk_size: int = 32,
) -> Tensor:
    """Return raw [N] text-to-visual attention, averaged over questions/heads.

    Every question row normalizes over all causally visible text+visual keys
    before visual columns are gathered. Causality uses sequence indices, while
    RoPE uses the supplied original position IDs. No visual-only renormalizing,
    Min-Max scaling, hidden-state forwarding or KV-cache reuse is performed.

    Invalid interfaces/data raise ProbeUnavailable for full-keep fallback.
    Runtime failures such as CUDA OOM deliberately propagate to the caller.
    """
    if not isinstance(inputs_embeds, Tensor) or inputs_embeds.ndim != 3:
        raise ProbeUnavailable("inputs_embeds must have shape [1, L, D]")
    batch, length, width = inputs_embeds.shape
    if batch != 1 or length == 0 or width == 0:
        raise ProbeUnavailable("Probe requires one nonempty, unpadded sequence")
    if inputs_embeds.dtype not in (torch.float16, torch.float32):
        raise ProbeUnavailable("Probe accepts FP16 inference or FP32 reference inputs")
    if not isinstance(position_ids, Tensor) or position_ids.shape != (1, length):
        raise ProbeUnavailable("position_ids must have shape [1, L]")
    if position_ids.dtype != torch.long or position_ids.device != inputs_embeds.device:
        raise ProbeUnavailable("position_ids must be long indices on the input device")
    # Native 4.37.2 prefill builds its rotary lookup with seq_len=L. Gapped
    # post-pruning IDs and cache offsets are outside this unpruned Probe API.
    if bool((position_ids < 0).any()) or bool((position_ids >= length).any()):
        raise ProbeUnavailable("Original prefill position IDs must be within [0, L)")
    if type(query_chunk_size) is not int or query_chunk_size <= 0:
        raise ProbeUnavailable("query_chunk_size must be a positive integer")
    _validate_indices("question_positions", question_positions, length)
    _validate_indices("visual_positions", visual_positions, length)
    if set(question_positions).intersection(visual_positions):
        raise ProbeUnavailable("Question text positions must not overlap visual positions")
    _require_finite("inputs_embeds", inputs_embeds)

    layer, attention = _native_first_layer(model)
    heads = attention.num_heads
    kv_heads = attention.num_key_value_heads
    head_dim = attention.head_dim
    groups = attention.num_key_value_groups
    if (
        min(heads, kv_heads, head_dim, groups) <= 0
        or heads != kv_heads * groups
        or width != heads * head_dim
        or attention.hidden_size != width
        or tuple(attention.q_proj.weight.shape) != (heads * head_dim, width)
        or tuple(attention.k_proj.weight.shape) != (kv_heads * head_dim, width)
        or tuple(layer.input_layernorm.weight.shape) != (width,)
    ):
        raise ProbeUnavailable("Input width or native Llama head/projection shapes do not match")
    for component in (layer.input_layernorm, attention.q_proj, attention.k_proj):
        if component.weight.device != inputs_embeds.device or component.weight.dtype != inputs_embeds.dtype:
            raise ProbeUnavailable("Probe inputs must match shared first-layer weight device/dtype")
    rotary = attention.rotary_emb
    if length > rotary.max_seq_len_cached:
        # Native rotary.forward extends registered buffers beyond the cache.
        # A read-only Probe must not mutate the shared model in that case.
        raise ProbeUnavailable("Probe sequence exceeds the existing native RoPE cache")
    if rotary.cos_cached.device != inputs_embeds.device or rotary.sin_cached.device != inputs_embeds.device:
        raise ProbeUnavailable("Native RoPE buffers must already be on the input device")

    normalized = layer.input_layernorm(inputs_embeds)
    _require_finite("input_layernorm output", normalized)
    keys = attention.k_proj(normalized).view(1, length, kv_heads, head_dim).transpose(1, 2)
    _require_finite("key projection", keys)
    cos, sin = rotary(keys, seq_len=length)
    # The native helper indexes both arguments with the same position_ids.
    # Rotate full K and chunked Q separately so each uses its original IDs.
    keys, _ = apply_rotary_pos_emb(keys, keys, cos, sin, position_ids)
    keys = repeat_kv(keys, groups)
    _require_finite("rotated keys", keys)
    key_transpose = keys.float().transpose(2, 3)

    question_indices = torch.tensor(question_positions, dtype=torch.long, device=inputs_embeds.device)
    visual_indices = torch.tensor(visual_positions, dtype=torch.long, device=inputs_embeds.device)
    key_indices = torch.arange(length, device=inputs_embeds.device)
    relevance_sum = torch.zeros(len(visual_positions), dtype=torch.float32, device=inputs_embeds.device)
    scale = math.sqrt(head_dim)

    for start in range(0, len(question_positions), query_chunk_size):
        chunk_indices = question_indices[start : start + query_chunk_size]
        chunk_length = chunk_indices.numel()
        queries = attention.q_proj(normalized.index_select(1, chunk_indices))
        queries = queries.view(1, chunk_length, heads, head_dim).transpose(1, 2)
        queries, _ = apply_rotary_pos_emb(
            queries, queries, cos, sin, position_ids.index_select(1, chunk_indices)
        )
        _require_finite("rotated question queries", queries)
        # Scores and softmax stay FP32 even when shared weights are FP16.
        # Memory scales with H * chunk_size * L, not with H * L * L.
        scores = torch.matmul(queries.float(), key_transpose) / scale
        _require_finite("question-to-sequence scores", scores)
        future_keys = key_indices[None, :] > chunk_indices[:, None]
        scores.masked_fill_(future_keys[None, None, :, :], float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        _require_finite("question-to-sequence probabilities", probabilities)
        relevance_sum += probabilities.index_select(-1, visual_indices).sum(dim=(0, 1, 2))

    relevance = relevance_sum / (heads * len(question_positions))
    _require_finite("raw visual relevance", relevance)
    return relevance
