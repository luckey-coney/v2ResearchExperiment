"""Timing contracts for later execution; no test/model ran during this edit."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from v2_pruning.llava_adapter import LlavaAdapter, PreparedInput
from v2_pruning.timing import _StageTimer


class TimingContractTests(unittest.TestCase):
    def test_cpu_timer_accumulates_even_on_error_and_none_disables_clock(self):
        timings = {}
        with patch("v2_pruning.timing.time.perf_counter", side_effect=[1.0, 1.002, 2.0, 2.003]):
            with _StageTimer(timings, "stage", torch.device("cpu")):
                pass
            with self.assertRaisesRegex(ValueError, "fixture"):
                with _StageTimer(timings, "stage", torch.device("cpu")):
                    raise ValueError("fixture")
        self.assertAlmostEqual(timings["stage"], 5.0)
        with patch("v2_pruning.timing.time.perf_counter") as clock:
            with _StageTimer(None, "stage", torch.device("cpu")):
                pass
            clock.assert_not_called()

    def fixture(self):
        config = LlamaConfig(vocab_size=24, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=64,
                             bos_token_id=1, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "eager"
        adapter = object.__new__(LlavaAdapter)
        adapter.model = LlamaForCausalLM(config).eval()
        adapter.torch, adapter.device = torch, torch.device("cpu")
        adapter.context_length, adapter.synchronize = 64, lambda: None
        adapter.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2,
                                            decode=lambda ids, **kwargs: "fixture")
        ids = torch.tensor([[1, 3, 1, 1, 5, 6]])
        embeddings = adapter.model.get_input_embeddings()(ids).detach()
        prepared = PreparedInput(
            embeddings[0, 2:4], embeddings, ids, torch.arange(6).unsqueeze(0),
            [2, 3], [4, 5], (1, 2), None, (2, 1), "fixture",
        )
        return adapter, prepared

    def test_native_generation_times_one_prefill_and_each_later_decoder_forward(self):
        adapter, prepared = self.fixture()
        stages = []

        class CaptureTimer(_StageTimer):
            def __enter__(timer):
                stages.append(timer.name)
                return super().__enter__()

        with patch("v2_pruning.llava_adapter._StageTimer", CaptureTimer):
            result = adapter.generate(prepared, [0], 3)
        self.assertEqual(stages.count("generation_prefill_ms"), 1)
        self.assertEqual(stages.count("generation_decode_forward_ms"), 2)
        self.assertGreaterEqual(result["generation_prefill_ms"], 0)
        self.assertGreaterEqual(result["generation_decode_forward_ms"], 0)
        self.assertIsNone(result["generation_decode_ms"])
        self.assertEqual(result["generated_tokens"], 3)

    def test_generation_failure_propagates_and_removes_its_hooks(self):
        adapter, prepared = self.fixture()
        decoder = adapter.model.model
        before = (len(decoder._forward_pre_hooks), len(decoder._forward_hooks))
        with patch.object(decoder, "forward", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                adapter.generate(prepared, [0], 3)
        self.assertEqual((len(decoder._forward_pre_hooks), len(decoder._forward_hooks)), before)


if __name__ == "__main__":
    unittest.main()
