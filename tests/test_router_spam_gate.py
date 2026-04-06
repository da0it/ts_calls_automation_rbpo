from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT / "services" / "router"
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

try:
    from routing.ai_analyzer import RubertEmbeddingAnalyzer
    from routing.models import CallInput, Segment
    from routing.nlp_preprocess import PreprocessConfig

    _ROUTER_TESTS_AVAILABLE = True
except Exception:
    RubertEmbeddingAnalyzer = None
    CallInput = None
    Segment = None
    PreprocessConfig = None
    _ROUTER_TESTS_AVAILABLE = False


class _FakeSpamGate:
    def __init__(self, *, status: str, confidence: float = 0.9) -> None:
        self.status = status
        self.confidence = confidence

    def predict(self, text: str, *, skip: bool = False):
        return {
            "active": True,
            "enabled": True,
            "predicted_label": "spam",
            "positive_label": "spam",
            "positive_confidence": self.confidence,
            "threshold": 0.85,
            "allow_threshold": 0.60,
            "backend": "sklearn_tfidf",
        }

    def build_decision(self, spam_meta):
        return {
            "status": self.status,
            "predicted_label": "spam",
            "confidence": self.confidence,
            "reason": f"fake_{self.status}",
            "backend": "sklearn_tfidf",
        }

    def status(self):
        return {"active": True}

    def reload_from_disk(self):
        return None


@unittest.skipUnless(_ROUTER_TESTS_AVAILABLE, "router ML dependencies are not installed")
class RouterSpamGateTest(unittest.TestCase):
    def _make_analyzer(self) -> RubertEmbeddingAnalyzer:
        return RubertEmbeddingAnalyzer(
            preprocess_cfg=PreprocessConfig(
                backend="none",
                model_text_mode="plain",
                drop_fillers=False,
                dedupe=False,
                keep_timestamps=False,
                do_tokenize=False,
                do_lemmatize=False,
            ),
            finetuned_enabled=True,
            spam_gate_enabled=True,
        )

    def _call(self) -> CallInput:
        return CallInput(
            call_id="call-1",
            segments=[Segment(start=0.0, end=1.0, speaker="speaker_0", role=None, text="хочу консультацию по продукту")],
        )

    def test_spam_block_stays_spam(self) -> None:
        analyzer = self._make_analyzer()
        analyzer._spam_gate = _FakeSpamGate(status="block", confidence=0.92)

        allowed_intents = {
            "spam.call": {"default_group": "support", "priority": "high"},
            "consulting": {"default_group": "consulting", "priority": "medium"},
            "misc.triage": {"default_group": "support", "priority": "medium"},
        }

        result = analyzer.analyze(self._call(), allowed_intents)

        self.assertEqual(result.intent.intent_id, "spam.call")
        self.assertEqual(result.raw["spam_decision"]["status"], "block")

    def test_spam_review_returns_spam_intent(self) -> None:
        analyzer = self._make_analyzer()
        analyzer._spam_gate = _FakeSpamGate(status="review", confidence=0.70)

        allowed_intents = {
            "spam.call": {"default_group": "support", "priority": "high"},
            "consulting": {"default_group": "consulting", "priority": "medium"},
            "misc.triage": {"default_group": "support", "priority": "medium"},
        }

        result = analyzer.analyze(self._call(), allowed_intents)

        self.assertEqual(result.intent.intent_id, "spam.call")
        self.assertEqual(result.raw["spam_decision"]["status"], "review")


if __name__ == "__main__":
    unittest.main()
