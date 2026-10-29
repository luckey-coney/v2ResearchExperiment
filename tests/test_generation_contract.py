"""Tiny randomly initialized CPU model contract; no weight downloads.

Written for later execution. No test/model was executed during implementation.
"""
from types import SimpleNamespace
import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from v2_pruning.llava_adapter import LlavaAdapter, PreparedInput


class GenerationContractTests(unittest.TestCase):
    def test_actual_short_sequence_native_positions_and_continuation(self):
        torch.manual_seed(7)
        config = LlamaConfig(vocab_size=24, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=64,
                             bos_token_id=1, eos_token_id=None, pad_token_id=0)
        config._attn_implementation = "eager"
        model = LlamaForCausalLM(config).eval()
        adapter = object.__new__(LlavaAdapter)
        adapter.model, adapter.torch, adapter.device = model, torch, torch.device("cpu")
        adapter.context_length = 64
        adapter.synchronize = lambda: None
        adapter.tokenizer = SimpleNamespace(
            pad_token_id=0, eos_token_id=2,
            decode=lambda ids, **kwargs: ",".join(str(int(i)) for i in ids),
        )
        ids = torch.tensor([[1, 3, 1, 1, 1, 1, 5, 6, 7, 8]])
        embeddings = model.get_input_embeddings()(ids).detach()
        prepared = PreparedInput(
            v0=embeddings[0, 2:6], inputs_embeds=embeddings, token_ids=ids,
            position_ids=torch.arange(10).unsqueeze(0), visual_positions=[2, 3, 4, 5],
            question_positions=[6, 7, 8], grid_shape=(2, 2), pixel_values=None,
            original_size=(2, 2), prompt="synthetic contract fixture",
        )
        calls = []

        def capture(module, args, kwargs):
            calls.append({key: value.detach().clone() if torch.is_tensor(value) else value
                          for key, value in kwargs.items() if key in {"inputs_embeds", "position_ids"}})

        hook = model.model.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            result = adapter.generate(prepared, [1, 3], 2)
        finally:
            hook.remove()
        keep = [0, 1, 3, 5, 6, 7, 8, 9]
        self.assertEqual(result["input_sequence_length"], 8)
        self.assertEqual(result["generated_tokens"], 2)
        self.assertEqual(len(result["answer"].split(",")), 2)
        self.assertTrue(torch.equal(calls[0]["inputs_embeds"], embeddings[:, keep]))
        self.assertTrue(torch.equal(calls[0]["position_ids"], torch.arange(8).unsqueeze(0)))
        self.assertTrue(torch.equal(calls[1]["position_ids"], torch.tensor([[8]])))
        self.assertEqual(result["position_policy"], "compact")


if __name__ == "__main__":
    unittest.main()
