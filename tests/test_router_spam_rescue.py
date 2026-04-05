from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTER_DIR = ROOT / "services" / "router"
if str(ROUTER_DIR) not in sys.path:
    sys.path.insert(0, str(ROUTER_DIR))

try:
    import torch
    from routing.ai_analyzer import RubertEmbeddingAnalyzer
    from routing.models import CallInput, Segment
    from routing.nlp_preprocess import PreprocessConfig
    _ROUTER_TESTS_AVAILABLE = True
except Exception:
    torch = None
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


class _FakeFinetunedRouter:
    def __init__(self, probs: torch.Tensor | None, intent_ids: list[str]) -> None:
        self.probs = probs
        self.intent_ids = intent_ids

    def predict(self, text: str, runtime_intent_ids: list[str]):
        if self.probs is None:
            return None, {"active": False, "reason": "fake_unavailable"}
        return self.probs, {"active": True, "intent_ids": self.intent_ids}

    def status(self, *, current_intents=None):
        return {"active": True}

    def reload_from_disk(self):
        return None


@unittest.skipUnless(_ROUTER_TESTS_AVAILABLE, "router ML dependencies are not installed")
class RouterSpamRescueTest(unittest.TestCase):
    def _make_analyzer(self, **kwargs) -> RubertEmbeddingAnalyzer:
        analyzer = RubertEmbeddingAnalyzer(
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
            **kwargs,
        )
        return analyzer

    def _call(self) -> CallInput:
        return CallInput(
            call_id="call-1",
            segments=[Segment(start=0.0, end=1.0, speaker="speaker_0", role=None, text="хочу консультацию по продукту")],
        )

    def test_stage2_can_rescue_false_spam_block(self) -> None:
        analyzer = self._make_analyzer()
        analyzer._spam_gate = _FakeSpamGate(status="block", confidence=0.92)
        analyzer._finetuned_router = _FakeFinetunedRouter(
            probs=torch.tensor([0.72, 0.28], dtype=torch.float32),
            intent_ids=["consulting", "portal_access"],
        )

        allowed_intents = {
            "spam.call": {"default_group": "support", "priority": "high"},
            "consulting": {"default_group": "consulting", "priority": "medium"},
            "portal_access": {"default_group": "support", "priority": "high"},
            "misc.triage": {"default_group": "support", "priority": "medium"},
        }

        result = analyzer.analyze(self._call(), allowed_intents)

        self.assertEqual(result.intent.intent_id, "consulting")
        self.assertEqual(result.raw["spam_decision"]["status"], "allow")
        self.assertIn("rescued_by_stage2", result.raw["spam_decision"]["reason"])

    def test_spam_review_no_longer_returns_triage_intent(self) -> None:
        analyzer = self._make_analyzer(spam_rescue_enabled=False)
        analyzer._spam_gate = _FakeSpamGate(status="review", confidence=0.70)
        analyzer._finetuned_router = _FakeFinetunedRouter(
            probs=torch.tensor([0.40, 0.60], dtype=torch.float32),
            intent_ids=["consulting", "portal_access"],
        )

        allowed_intents = {
            "spam.call": {"default_group": "support", "priority": "high"},
            "consulting": {"default_group": "consulting", "priority": "medium"},
            "portal_access": {"default_group": "support", "priority": "high"},
            "misc.triage": {"default_group": "support", "priority": "medium"},
        }

        result = analyzer.analyze(self._call(), allowed_intents)

        self.assertEqual(result.intent.intent_id, "spam.call")


if __name__ == "__main__":
    unittest.main()
