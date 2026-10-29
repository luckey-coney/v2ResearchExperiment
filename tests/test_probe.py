"""Offline, no-download Probe checks; written but not executed in this task.

Reference: transformers v4.37.2, commit
345b9b1a6a308a1fa6559251eb33ead2211240ac,
src/transformers/models/llama/modeling_llama.py:LlamaDecoderLayer.
Only a tiny native decoder layer is constructed; no language model is loaded.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import v2_pruning.probe as probe_module
from v2_pruning.probe import ProbeUnavailable, probe_relevance


class _TinyLlavaInterface(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([layer])

    def get_model(self):
        return self.decoder


def _case(num_key_value_heads=4):
    torch.manual_seed(17)
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=32,
        attention_dropout=0.0,
        pretraining_tp=1,
    )
    config._attn_implementation = "eager"
    model = _TinyLlavaInterface(LlamaDecoderLayer(config, layer_idx=0)).eval()
    embeds = torch.randn(1, 10, config.hidden_size)
    positions = torch.arange(10, dtype=torch.long).unsqueeze(0)
    return model, embeds, positions


@torch.inference_mode()
def _native_relevance(model, embeds, questions, visuals, positions):
    """Full native decoder attention is the oracle, not a second Q/K formula."""
    length = embeds.shape[1]
    causal_mask = torch.full(
        (length, length),
        torch.finfo(embeds.dtype).min,
        dtype=embeds.dtype,
        device=embeds.device,
    ).triu(diagonal=1)[None, None]
    outputs = model.get_model().layers[0](
        hidden_states=embeds,
        attention_mask=causal_mask,
        position_ids=positions,
        output_attentions=True,
        use_cache=False,
    )
    attention = outputs[1][0].float()
    question_indices = torch.tensor(questions, device=embeds.device)
    visual_indices = torch.tensor(visuals, device=embeds.device)
    return (
        attention.index_select(1, question_indices)
        .index_select(2, visual_indices)
        .mean(dim=(0, 1))
    )


@pytest.mark.parametrize("num_key_value_heads", [1, 2, 4])
def test_matches_full_native_first_layer_attention_including_gqa(num_key_value_heads):
    model, embeds, positions = _case(num_key_value_heads)
    questions, visuals = [3, 6, 8, 9], [1, 2, 4]
    expected = _native_relevance(model, embeds, questions, visuals, positions)

    actual = probe_relevance(model, embeds, questions, visuals, positions, 2)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    assert actual.dtype == torch.float32
    assert not actual.requires_grad


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 32])
def test_question_chunk_sizes_preserve_result(chunk_size):
    model, embeds, positions = _case()
    questions, visuals = [3, 6, 8, 9], [1, 2, 4]
    expected = probe_relevance(model, embeds, questions, visuals, positions, 4)

    actual = probe_relevance(model, embeds, questions, visuals, positions, chunk_size)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)


def test_only_requested_question_rows_are_projected_in_bounded_chunks():
    model, embeds, positions = _case()
    attention = model.get_model().layers[0].self_attn
    query_lengths, key_lengths = [], []
    query_hook = attention.q_proj.register_forward_pre_hook(
        lambda module, args: query_lengths.append(args[0].shape[1])
    )
    key_hook = attention.k_proj.register_forward_pre_hook(
        lambda module, args: key_lengths.append(args[0].shape[1])
    )
    try:
        probe_relevance(model, embeds, [3, 6, 8, 9], [1, 2, 4], positions, 3)
    finally:
        query_hook.remove()
        key_hook.remove()

    assert query_lengths == [3, 1]
    assert key_lengths == [embeds.shape[1]]


def test_rope_uses_original_position_ids_but_causality_uses_sequence_indices():
    model, embeds, _ = _case()
    positions = torch.tensor([[5, 1, 7, 0, 9, 2, 4, 8, 3, 6]])
    questions, visuals = [0, 3, 8, 9], [1, 2, 4]
    expected = _native_relevance(model, embeds, questions, visuals, positions)

    actual = probe_relevance(model, embeds, questions, visuals, positions, 2)

    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
    early_query = probe_relevance(model, embeds, [0], visuals, positions)
    torch.testing.assert_close(early_query, torch.zeros_like(early_query))


def test_softmax_keeps_text_keys_and_does_not_renormalize_visual_columns():
    model, embeds, positions = _case()
    attention = model.get_model().layers[0].self_attn
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.k_proj.weight.zero_()

    actual = probe_relevance(model, embeds, [8, 9], [1, 2, 4], positions)

    # Both rows have uniform logits. There are respectively 9 and 10 legal
    # text+visual keys; each visual therefore receives their mean probability.
    expected = torch.full((3,), (1.0 / 9.0 + 1.0 / 10.0) / 2.0)
    visual_only_softmax = torch.full((3,), 1.0 / 3.0)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, visual_only_softmax)
    assert actual.sum().item() < 1.0


def test_visual_result_order_matches_requested_original_indices():
    model, embeds, positions = _case()
    ordered = probe_relevance(model, embeds, [8, 9], [1, 2, 4], positions)

    permuted = probe_relevance(model, embeds, [8, 9], [4, 1, 2], positions)

    torch.testing.assert_close(permuted, ordered[[2, 0, 1]])


def test_does_not_call_decoder_v_projection_mlp_or_output_projection(monkeypatch):
    model, embeds, positions = _case()
    layer = model.get_model().layers[0]

    def forbidden(*args, **kwargs):
        raise AssertionError("Probe must only borrow input LN and Q/K components")

    for module in (layer, layer.self_attn, layer.self_attn.v_proj, layer.mlp, layer.self_attn.o_proj):
        monkeypatch.setattr(module, "forward", forbidden)

    actual = probe_relevance(model, embeds, [8, 9], [1, 2, 4], positions)

    assert actual.shape == (3,)


def test_shared_parameters_buffers_and_input_remain_unchanged():
    model, embeds, positions = _case()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    parameter_ids = {name: id(value) for name, value in model.named_parameters()}
    requires_grad = {name: value.requires_grad for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    original_embeds, original_positions = embeds.clone(), positions.clone()

    probe_relevance(model, embeds, [8, 9], [1, 2, 4], positions)

    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        assert id(value) == parameter_ids[name]
        assert value.requires_grad == requires_grad[name]
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    torch.testing.assert_close(embeds, original_embeds, rtol=0, atol=0)
    torch.testing.assert_close(positions, original_positions, rtol=0, atol=0)
    assert not model.training


@pytest.mark.parametrize(
    "questions,visuals",
    [
        ([], [1, 2]),
        ([8], []),
        ([-1], [1, 2]),
        ([10], [1, 2]),
        ([8, 8], [1, 2]),
        ([8], [1, 1]),
        ([8], [-1, 2]),
        ([8], [1, 10]),
        ([8], [1, 8]),
        ([True], [1, 2]),
        ([8], [1.0, 2]),
        ((8,), [1, 2]),
    ],
)
def test_empty_invalid_duplicate_or_overlapping_positions_are_unavailable(questions, visuals):
    model, embeds, positions = _case()
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, questions, visuals, positions)


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_sizes_are_unavailable(chunk_size):
    model, embeds, positions = _case()
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions, chunk_size)


@pytest.mark.parametrize(
    "invalid_case",
    [
        "batch", "rank", "width", "dtype", "positions_shape",
        "positions_dtype", "positions_negative", "positions_outside_sequence",
    ],
)
def test_invalid_tensor_contracts_are_unavailable(invalid_case):
    model, embeds, positions = _case()
    if invalid_case == "batch":
        embeds = embeds.expand(2, -1, -1)
    elif invalid_case == "rank":
        embeds = embeds[0]
    elif invalid_case == "width":
        embeds = embeds[:, :, :-1]
    elif invalid_case == "dtype":
        embeds = embeds.double()
    elif invalid_case == "positions_shape":
        positions = positions[0]
    elif invalid_case == "positions_dtype":
        positions = positions.float()
    elif invalid_case == "positions_negative":
        positions[0, 0] = -1
    else:
        positions[0, -1] = embeds.shape[1]

    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_input_is_unavailable(nonfinite):
    model, embeds, positions = _case()
    embeds[0, 0, 0] = nonfinite
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)


def test_nonfinite_projected_values_are_unavailable():
    model, embeds, positions = _case()
    with torch.no_grad():
        model.get_model().layers[0].self_attn.q_proj.weight[0, 0] = float("nan")
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)


def test_unsupported_model_and_transformers_version_are_unavailable(monkeypatch):
    model, embeds, positions = _case()
    with pytest.raises(ProbeUnavailable):
        probe_relevance(SimpleNamespace(), embeds, [8], [1, 2], positions)
    monkeypatch.setattr(probe_module.transformers, "__version__", "4.55.0")
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)


def test_qwen_architecture_is_explicitly_unavailable():
    model, embeds, positions = _case()
    model.get_model().layers[0].self_attn.config.model_type = "qwen2"
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)


def test_rope_cache_is_not_extended_by_probe():
    model, embeds, positions = _case()
    rotary = model.get_model().layers[0].self_attn.rotary_emb
    rotary.max_seq_len_cached = 5
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)
    assert rotary.max_seq_len_cached == 5


def test_pretraining_tensor_parallel_variant_is_explicitly_unavailable():
    model, embeds, positions = _case()
    model.get_model().layers[0].self_attn.config.pretraining_tp = 2
    with pytest.raises(ProbeUnavailable):
        probe_relevance(model, embeds, [8], [1, 2], positions)


def test_out_of_memory_is_not_swallowed_as_full_keep_fallback(monkeypatch):
    model, embeds, positions = _case()

    def out_of_memory(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic OOM; no CUDA allocation")

    monkeypatch.setattr(model.get_model().layers[0].self_attn.q_proj, "forward", out_of_memory)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        probe_relevance(model, embeds, [8], [1, 2], positions)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 native reference needs CUDA")
def test_fp16_gpu_probe_matches_native_attention_with_rounding_tolerance():
    model, embeds, positions = _case(2)
    model = model.to(device="cuda", dtype=torch.float16)
    embeds, positions = embeds.to(device="cuda", dtype=torch.float16), positions.cuda()
    questions, visuals = [3, 6, 8, 9], [1, 2, 4]
    expected = _native_relevance(model, embeds, questions, visuals, positions)

    actual = probe_relevance(model, embeds, questions, visuals, positions, 2)

    # Native eager attention rounds the QK product and probabilities to FP16;
    # Probe uses FP32 logits/softmax to avoid an overflow-based false fallback.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-4)
