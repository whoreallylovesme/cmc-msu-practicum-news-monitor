import sys
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "ml"))
sys.path.insert(0, str(ROOT))

import qwen_ensemble


class TinyTokenizer:
    pad_token = None
    pad_token_id = 0
    eos_token = "<|endoftext|>"


class QwenBackendTests(unittest.TestCase):
    def test_auto_device_loads_tiny_qwen_without_downloads(self):
        expected = ("cuda" if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu")
        with tempfile.TemporaryDirectory() as directory:
            config = Qwen3Config(
                vocab_size=64, hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=32, tie_word_embeddings=True,
            )
            model = AutoModelForCausalLM.from_config(config)
            model.save_pretrained(directory)
            artifact = {"assets": directory, "device": "auto"}
            model = qwen_ensemble.load_model(artifact, TinyTokenizer())
            self.assertEqual(next(model.parameters()).device.type, expected)
            input_ids = torch.tensor([[1, 2, 3, 4]], device=next(model.parameters()).device)
            with torch.inference_mode():
                logits = model(input_ids=input_ids).logits
            self.assertEqual(logits.shape, (1, 4, 64))


if __name__ == "__main__":
    unittest.main()
