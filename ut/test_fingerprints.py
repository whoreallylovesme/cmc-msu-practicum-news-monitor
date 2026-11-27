"""Shared fingerprint helpers must preserve the persisted digest contracts."""

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "runtime/storage"), str(ROOT / "runtime/ml")]

from lib.fingerprints import comment_content_hash, label_context_fingerprint
from demo.e2e import context_fingerprint
from runtime.ml.dataset_labels import fingerprint as dataset_label_fingerprint
from runtime.ml.training_data import fingerprint as training_fingerprint
from runtime.storage.main import Comment


ITEM = {
    "id": "comment-1",
    "story_id": "story-ю",
    "story_title": "Заголовок 🚀",
    "parent_id": None,
    "parent_text": "Родитель\nтекст",
    "text": "Комментарий café",
    "published_at": "2026-01-02T03:04:05Z",
    "url": "https://example.test/comment",
    "source": "hacker_news",
}
COMMENT_CONTENT_GOLDEN = "c50297b8bfbb7e63300abb16e5b6f5df62822c7d92409731351d75b2aef034ae"
LABEL_CONTEXT_GOLDEN = "d505435989025003baa189975689add348d5964372c37f0670b5af1af2938296"


class FingerprintGoldenTests(unittest.TestCase):
    def test_comment_content_hash_contract(self):
        implementations = {
            "common": comment_content_hash(ITEM),
            "storage": Comment(**ITEM).storage_row("ChatGPT")["content_hash"],
            "training data": training_fingerprint(ITEM),
        }
        for name, digest in implementations.items():
            with self.subTest(implementation=name):
                self.assertEqual(digest, COMMENT_CONTENT_GOLDEN)

    def test_label_context_fingerprint_contract(self):
        implementations = {
            "common": label_context_fingerprint(ITEM),
            "dataset labels": dataset_label_fingerprint(ITEM),
            "demo": context_fingerprint(ITEM),
        }
        for name, digest in implementations.items():
            with self.subTest(implementation=name):
                self.assertEqual(digest, LABEL_CONTEXT_GOLDEN)


if __name__ == "__main__":
    unittest.main()
