import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
sys.path.insert(0, str(ROOT))

from backends import causal_lm_ensemble


class TinyTokenizer:
    pad_token = None
    pad_token_id = 0
    eos_token = "<|endoftext|>"


class CausalLmBackendTests(unittest.TestCase):
    def test_loads_tiny_auto_config_model_without_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            config = AutoConfig.for_model(
                "gpt2", vocab_size=64, n_positions=32, n_ctx=32,
                n_embd=32, n_layer=1, n_head=4,
            )
            model = AutoModelForCausalLM.from_config(config)
            model.save_pretrained(directory)
            del model
            artifact = {"assets": directory, "device": "cpu"}
            model = causal_lm_ensemble.load_model(artifact, TinyTokenizer())
            self.assertEqual(next(model.parameters()).device.type, "cpu")
            input_ids = torch.tensor([[1, 2, 3, 4]])
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits
            self.assertEqual(logits.shape, (1, 4, 64))
            causal_lm_ensemble.release(model)


if __name__ == "__main__":
    unittest.main()
