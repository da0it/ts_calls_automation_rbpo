from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import joblib
import torch


logger = logging.getLogger(__name__)


class SpamGateRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        model_path: str,
        artifact_path: str,
        threshold: float,
        allow_threshold: float,
        score_threshold: Optional[float],
        score_allow_threshold: Optional[float],
        positive_label: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_path = str(model_path or "").strip()
        self.artifact_path = str(artifact_path or "").strip()
        self.threshold = float(max(0.0, min(1.0, threshold)))
        self.allow_threshold = float(max(0.0, min(self.threshold, allow_threshold)))
        self.score_threshold = float(score_threshold) if score_threshold is not None else None
        self.score_allow_threshold = float(score_allow_threshold) if score_allow_threshold is not None else None
        if self.score_threshold is not None and self.score_allow_threshold is not None:
            self.score_allow_threshold = min(self.score_allow_threshold, self.score_threshold)
        self.positive_label = str(positive_label or "").strip() or "spam"

        self._state_lock = RLock()
        self._artifact: Optional[Dict[str, Any]] = None
        self._model: Optional[Any] = None
        self._vectorizer: Optional[Any] = None
        self._active_model_path: str = ""

        self.reload_from_disk()

    def reload_from_disk(self) -> None:
        if not self.artifact_path:
            self._clear_artifact()
            return
        path = Path(self.artifact_path)
        if not path.exists():
            self._clear_artifact()
            logger.info("No spam gate artifact found at %s", path)
            return
        try:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError("invalid spam gate artifact payload")
            self._activate_artifact(payload)
            logger.info("Loaded spam gate artifact from %s", path)
        except Exception as exc:
            self._clear_artifact()
            logger.warning("Failed to load spam gate artifact from %s: %s", path, exc)

    def predict(self, text: str, *, skip: bool = False) -> Dict[str, Any]:
        with self._state_lock:
            artifact = dict(self._artifact or {})

        if skip:
            return {
                "enabled": self.enabled,
                "active": False,
                "skipped": True,
                "reason": "skip_spam_gate_requested",
                "backend": self.backend(artifact),
            }
        if not self.enabled:
            return {"enabled": False, "active": False, "reason": "spam_gate_disabled", "backend": self.backend(artifact)}
        if not artifact:
            return {"enabled": True, "active": False, "reason": "no_spam_gate_model", "backend": self.backend(artifact)}

        gate_meta = artifact.get("spam_gate")
        if not isinstance(gate_meta, dict) or not gate_meta.get("enabled"):
            return {"enabled": True, "active": False, "reason": "spam_gate_not_enabled", "backend": self.backend(artifact)}

        backend = self.backend(artifact)
        if backend != "sklearn_tfidf":
            return {"enabled": True, "active": False, "reason": f"unsupported_backend:{backend}", "backend": backend}

        labels = self.labels(artifact)
        positive_label = str(gate_meta.get("positive_label") or self.positive_label).strip() or self.positive_label
        if positive_label not in labels:
            return {"enabled": True, "active": False, "reason": "positive_label_missing", "backend": backend}

        try:
            probs, model_path, raw_score = self._predict_with_sklearn(text=text, artifact=artifact, labels=labels)
            best_idx = int(torch.argmax(probs).item())
            positive_idx = labels.index(positive_label)
            positive_confidence = float(probs[positive_idx].item())
            return {
                "enabled": True,
                "active": True,
                "labels": labels,
                "positive_label": positive_label,
                "predicted_label": labels[best_idx],
                "positive_confidence": positive_confidence,
                "positive_score": raw_score,
                "threshold": float(gate_meta.get("threshold") or self.threshold),
                "allow_threshold": float(gate_meta.get("allow_threshold") or self.allow_threshold),
                "score_threshold": self.score_threshold_value(gate_meta),
                "score_allow_threshold": self.score_allow_threshold_value(gate_meta),
                "temperature": 1.0,
                "model_path": model_path or gate_meta.get("model_path", ""),
                "trained_at": gate_meta.get("trained_at", ""),
                "backend": backend,
            }
        except Exception as exc:
            logger.warning("Failed to run spam gate model: %s", exc)
            return {"enabled": True, "active": False, "reason": f"runtime_error:{exc}", "backend": backend}

    def build_decision(self, spam_meta: Dict[str, Any]) -> Dict[str, Any]:
        threshold_high = float(spam_meta.get("threshold") or self.threshold)
        threshold_low = float(spam_meta.get("allow_threshold") or self.allow_threshold)
        threshold_low = max(0.0, min(threshold_low, threshold_high))
        score_threshold_high = self._as_optional_float(spam_meta.get("score_threshold"))
        score_threshold_low = self._as_optional_float(spam_meta.get("score_allow_threshold"))
        if score_threshold_high is not None and score_threshold_low is not None:
            score_threshold_low = min(score_threshold_low, score_threshold_high)
        confidence = float(spam_meta.get("positive_confidence") or 0.0)
        raw_score = self._as_optional_float(spam_meta.get("positive_score"))
        predicted_label = str(spam_meta.get("predicted_label") or "").strip()
        positive_label = str(spam_meta.get("positive_label") or self.positive_label).strip() or self.positive_label

        status = "allow"
        reason = str(spam_meta.get("reason") or "ok")
        if spam_meta.get("skipped"):
            reason = "skip_spam_gate_requested"
        elif spam_meta.get("active"):
            if raw_score is not None and score_threshold_high is not None:
                if predicted_label == positive_label and raw_score >= score_threshold_high:
                    status = "block"
                    reason = f"positive_score>={score_threshold_high:.3f}"
                elif score_threshold_low is not None and raw_score <= score_threshold_low:
                    status = "allow"
                    reason = f"positive_score<={score_threshold_low:.3f}"
                elif confidence <= threshold_low:
                    status = "allow"
                    reason = f"positive_confidence<={threshold_low:.3f}"
                else:
                    status = "review"
                    if score_threshold_low is None:
                        reason = (
                            f"manual_review_required:score<{score_threshold_high:.3f}"
                            f",confidence>{threshold_low:.3f}"
                        )
                    else:
                        reason = (
                            f"manual_review_required:{score_threshold_low:.3f}"
                            f"<{raw_score:.3f}<{score_threshold_high:.3f}"
                        )
            elif predicted_label == positive_label and confidence >= threshold_high:
                status = "block"
                reason = f"positive_confidence>={threshold_high:.3f}"
            elif confidence <= threshold_low:
                status = "allow"
                reason = f"positive_confidence<={threshold_low:.3f}"
            else:
                status = "review"
                reason = f"manual_review_required:{threshold_low:.3f}<{confidence:.3f}<{threshold_high:.3f}"

        return {
            "status": status,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "score": raw_score or 0.0,
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "score_threshold_low": score_threshold_low or 0.0,
            "score_threshold_high": score_threshold_high or 0.0,
            "reason": reason,
            "skipped": bool(spam_meta.get("skipped")),
            "backend": str(spam_meta.get("backend") or self.backend({})),
        }

    def status(self) -> Dict[str, Any]:
        with self._state_lock:
            artifact = dict(self._artifact or {})
            active_model_path = self._active_model_path
        backend = self.backend(artifact)
        if not self.enabled:
            return {
                "enabled": False,
                "active": False,
                "threshold": self.threshold,
                "allow_threshold": self.allow_threshold,
                "score_threshold": self.score_threshold,
                "score_allow_threshold": self.score_allow_threshold,
                "reason": "spam_gate_disabled",
                "backend": backend,
            }
        if not artifact:
            return {
                "enabled": True,
                "active": False,
                "threshold": self.threshold,
                "allow_threshold": self.allow_threshold,
                "score_threshold": self.score_threshold,
                "score_allow_threshold": self.score_allow_threshold,
                "reason": "no_spam_gate_model",
                "model_path": self.model_path,
                "backend": backend,
            }
        gate = artifact.get("spam_gate") if isinstance(artifact.get("spam_gate"), dict) else {}
        labels = self.labels(artifact)
        active = bool(gate.get("enabled")) and len(labels) >= 2 and backend == "sklearn_tfidf"
        return {
            "enabled": True,
            "active": active,
            "reason": "ok" if active else ("unsupported_backend" if backend != "sklearn_tfidf" else "invalid_spam_gate_model"),
            "threshold": float(gate.get("threshold") or self.threshold),
            "allow_threshold": float(gate.get("allow_threshold") or self.allow_threshold),
            "score_threshold": self.score_threshold_value(gate),
            "score_allow_threshold": self.score_allow_threshold_value(gate),
            "positive_label": str(gate.get("positive_label") or self.positive_label),
            "labels": labels,
            "model_path": str(gate.get("model_path") or self.model_path),
            "trained_at": str(gate.get("trained_at") or ""),
            "calibration": self.model_calibration(gate),
            "backend": backend,
            "active_backend": backend if active_model_path else backend,
        }

    def labels(self, artifact: Dict[str, Any]) -> List[str]:
        labels = [str(x).strip() for x in list(artifact.get("labels") or []) if str(x).strip()]
        if labels:
            return labels
        gate = artifact.get("spam_gate")
        if isinstance(gate, dict):
            nested = [str(x).strip() for x in list(gate.get("labels") or []) if str(x).strip()]
            if nested:
                return nested
        return []

    def backend(self, artifact: Dict[str, Any]) -> str:
        gate = artifact.get("spam_gate") if isinstance(artifact.get("spam_gate"), dict) else {}
        backend = str(gate.get("backend") or artifact.get("backend") or "").strip().lower()
        if backend in {"sklearn", "svm_tfidf", "svm+tfidf", ""}:
            return "sklearn_tfidf"
        return backend

    def model_calibration(self, model_meta: Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(model_meta, dict) and isinstance(model_meta.get("calibration"), dict):
            return dict(model_meta.get("calibration") or {})
        return {}

    def score_threshold_value(self, model_meta: Dict[str, Any]) -> Optional[float]:
        value = self._as_optional_float(model_meta.get("score_threshold"))
        if value is not None:
            return value
        return self.score_threshold

    def score_allow_threshold_value(self, model_meta: Dict[str, Any]) -> Optional[float]:
        value = self._as_optional_float(model_meta.get("score_allow_threshold"))
        if value is not None:
            return value
        return self.score_allow_threshold

    def _as_optional_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or str(value).strip() == "":
                return None
            return float(value)
        except Exception:
            return None

    def _predict_with_sklearn(
        self,
        *,
        text: str,
        artifact: Dict[str, Any],
        labels: List[str],
    ) -> Tuple[torch.Tensor, str, Optional[float]]:
        model, vectorizer, model_path = self._ensure_sklearn_model_loaded(artifact)
        features: Any = [text]
        if vectorizer is not None:
            features = vectorizer.transform([text])
        raw_score: Optional[float] = None
        if hasattr(model, "predict_proba"):
            prob_list = model.predict_proba(features)[0]
        else:
            decision = model.decision_function(features)
            if hasattr(decision, "tolist"):
                decision = decision.tolist()
            if isinstance(decision, list) and decision and isinstance(decision[0], list):
                decision = decision[0]
            if not isinstance(decision, list):
                decision = [float(decision)]
            if len(decision) == 1:
                score = float(decision[0])
                raw_score = score
                prob_pos = 1.0 / (1.0 + math.exp(-score))
                prob_list = [1.0 - prob_pos, prob_pos]
            else:
                tensor = torch.tensor(decision, dtype=torch.float32)
                prob_list = torch.softmax(tensor, dim=0).tolist()
        probs = torch.tensor([float(x) for x in prob_list], dtype=torch.float32)
        if probs.numel() != len(labels):
            raise RuntimeError(f"spam gate classes mismatch: model={probs.numel()} labels={len(labels)}")
        return probs, model_path, raw_score

    def _ensure_sklearn_model_loaded(self, artifact: Dict[str, Any]) -> Tuple[Any, Any, str]:
        gate_meta = artifact.get("spam_gate")
        if not isinstance(gate_meta, dict):
            raise RuntimeError("spam gate metadata is missing")
        model_path_raw = str(gate_meta.get("model_path") or "").strip()
        if not model_path_raw:
            raise RuntimeError("spam gate model path is empty")

        model_dir = Path(model_path_raw)
        if model_dir.is_file():
            model_dir = model_dir.parent
        if not model_dir.exists():
            raise RuntimeError(f"spam gate model path not found: {model_dir}")

        model_file = self._resolve_model_file(
            model_dir,
            [
                gate_meta.get("classifier_file"),
                gate_meta.get("model_file"),
                "model.joblib",
                "classifier.joblib",
                "svm.joblib",
                "svm_model.joblib",
            ],
            "classifier/model",
        )
        vectorizer_file = self._resolve_model_file(
            model_dir,
            [
                gate_meta.get("vectorizer_file"),
                "vectorizer.joblib",
                "tfidf.joblib",
                "tfidf_vectorizer.joblib",
            ],
            "vectorizer",
            required=False,
        )

        with self._state_lock:
            if self._model is not None and self._active_model_path == str(model_dir):
                return self._model, self._vectorizer, str(model_dir)

            model = joblib.load(model_file)
            vectorizer = joblib.load(vectorizer_file) if vectorizer_file is not None else None

            self._model = model
            self._vectorizer = vectorizer
            self._active_model_path = str(model_dir)
            return model, vectorizer, str(model_dir)

    def _resolve_model_file(
        self,
        base_dir: Path,
        candidates: List[Any],
        label: str,
        required: bool = True,
    ) -> Optional[Path]:
        for candidate in candidates:
            name = str(candidate or "").strip()
            if not name:
                continue
            path = Path(name)
            if not path.is_absolute():
                path = base_dir / path
            if path.exists() and path.is_file():
                return path
        if required:
            raise RuntimeError(f"missing spam gate {label} artifact in {base_dir}")
        return None

    def _activate_artifact(self, artifact: Dict[str, Any]) -> None:
        with self._state_lock:
            self._artifact = dict(artifact)
            self._model = None
            self._vectorizer = None
            self._active_model_path = ""

    def _clear_artifact(self) -> None:
        with self._state_lock:
            self._artifact = None
            self._model = None
            self._vectorizer = None
            self._active_model_path = ""
