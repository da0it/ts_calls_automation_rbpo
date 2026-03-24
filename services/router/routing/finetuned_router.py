from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import time

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .finetuned_training import collect_training_samples, stratified_split, train_finetuned_model


logger = logging.getLogger(__name__)
RESERVED_FALLBACK_INTENT_ID = "misc.triage"


class FinetunedRouterRuntime:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        tuned_model_path: str,
        finetuned_enabled: bool,
        finetuned_model_path: str,
        finetuned_max_length: int,
        finetuned_weight_decay: float,
        max_text_chars: int,
    ) -> None:
        self.model_name = str(model_name).strip() or "ai-forever/ruBert-base"
        self.device = str(device).strip() or "cpu"
        self.tuned_model_path = str(tuned_model_path or "").strip()
        self.finetuned_enabled = bool(finetuned_enabled)
        self.finetuned_model_path = str(finetuned_model_path or "").strip()
        self.finetuned_max_length = int(max(64, min(512, finetuned_max_length)))
        self.finetuned_weight_decay = float(max(0.0, min(0.2, finetuned_weight_decay)))
        self.max_text_chars = int(max(200, min(20000, max_text_chars)))

        self._state_lock = RLock()
        self._artifact: Optional[Dict[str, Any]] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._tokenizer: Optional[Any] = None
        self._active_intents: Optional[Tuple[str, ...]] = None
        self._active_model_path: str = ""
        self._last_train_report: Optional[Dict[str, Any]] = None
        self._last_train_error: str = ""

        self.reload_from_disk()

    def reload_from_disk(self) -> None:
        if not self.tuned_model_path:
            self._clear_artifact()
            return
        path = Path(self.tuned_model_path)
        if not path.exists():
            self._clear_artifact()
            logger.info("No tuned router artifact found at %s", path)
            return
        try:
            payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise RuntimeError("invalid tuned artifact payload")
            self._activate_artifact(payload)
            logger.info("Loaded tuned router artifact from %s", path)
        except Exception as exc:
            self._clear_artifact()
            logger.warning("Failed to load tuned router artifact from %s: %s", path, exc)

    def predict(self, text: str, runtime_intent_ids: List[str]) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        if not self.finetuned_enabled:
            return None, {"active": False, "reason": "finetuned_disabled"}

        with self._state_lock:
            artifact = dict(self._artifact or {})
        if not artifact:
            return None, {"active": False, "reason": "no_tuned_model"}

        finetuned_meta = artifact.get("finetuned_model")
        if not isinstance(finetuned_meta, dict) or not finetuned_meta.get("enabled"):
            return None, {"active": False, "reason": "no_finetuned_model"}

        artifact_intents = self._artifact_intent_ids(artifact)
        if not self._same_intent_set(artifact_intents, runtime_intent_ids):
            return None, {
                "active": False,
                "reason": "intents_mismatch",
                "artifact_intents_n": len(artifact_intents),
                "runtime_intents_n": len(runtime_intent_ids),
            }

        try:
            model, tokenizer, max_len = self._ensure_model_loaded(artifact, artifact_intents)
            enc = tokenizer(
                [text],
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            temperature = self._artifact_temperature(artifact)
            with torch.inference_mode():
                logits = model(**enc).logits
                probs = torch.softmax(logits / temperature, dim=1).squeeze(0)
            return probs, {
                "active": True,
                "trained_at": finetuned_meta.get("trained_at", ""),
                "model_path": finetuned_meta.get("model_path", ""),
                "intent_ids": artifact_intents,
                "temperature": temperature,
            }
        except Exception as exc:
            logger.warning("Failed to run fine-tuned RuBERT head: %s", exc)
            return None, {"active": False, "reason": f"runtime_error:{exc}"}

    def status(self, *, current_intents: Optional[List[str]] = None) -> Dict[str, Any]:
        with self._state_lock:
            artifact = dict(self._artifact or {})
            report = dict(self._last_train_report or {})
            last_error = str(self._last_train_error or "")

        if not artifact:
            return {
                "active": False,
                "reason": "no_tuned_model",
                "model_path": self.tuned_model_path,
                "finetuned_model": {
                    "enabled": self.finetuned_enabled,
                    "active": False,
                    "model_path": self.finetuned_model_path,
                },
                "last_train_report": report,
                "last_train_error": last_error,
            }

        artifact_intents = self._artifact_intent_ids(artifact)
        compatible = current_intents is None or self._same_intent_set(artifact_intents, current_intents)
        order_matches = current_intents is not None and self._comparable_intent_ids(artifact_intents) == self._comparable_intent_ids(current_intents)

        finetuned_model = artifact.get("finetuned_model") if isinstance(artifact.get("finetuned_model"), dict) else {}
        finetuned_ready = bool(finetuned_model and finetuned_model.get("enabled"))
        active = compatible and finetuned_ready
        reason = "ok" if active else ("intents_mismatch" if not compatible else "no_finetuned_model")

        return {
            "active": active,
            "reason": reason,
            "model_path": self.tuned_model_path,
            "version_id": artifact.get("version_id", ""),
            "trained_at": artifact.get("trained_at", ""),
            "intent_ids": artifact_intents,
            "current_intents": current_intents,
            "intent_order_matches_current": order_matches,
            "metrics": artifact.get("metrics", {}),
            "dataset": artifact.get("dataset", {}),
            "finetuned_model": {
                "enabled": self.finetuned_enabled,
                "active": active,
                "model_path": finetuned_model.get("model_path", self.finetuned_model_path),
                "metrics": finetuned_model.get("metrics", {}),
                "calibration": self._artifact_calibration(artifact),
            },
            "last_train_report": report,
            "last_train_error": last_error,
        }

    def train(
        self,
        *,
        allowed_intents: Dict[str, Dict[str, Any]],
        runtime_intent_ids: List[str],
        feedback_path: str,
        output_path: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        val_ratio: float,
        random_seed: int,
    ) -> Dict[str, Any]:
        started = time.time()
        self._set_last_train_error("")
        try:
            if not self.finetuned_enabled:
                raise RuntimeError("fine-tuning is disabled (set ROUTER_FINETUNED_ENABLED=1)")

            samples, dataset_meta = collect_training_samples(
                allowed_intents=allowed_intents,
                feedback_path=feedback_path,
                max_text_chars=self.max_text_chars,
            )
            if not samples:
                raise RuntimeError("no training samples after preprocessing")

            label_to_idx = {iid: i for i, iid in enumerate(runtime_intent_ids)}
            filtered: List[Dict[str, Any]] = []
            for sample in samples:
                intent_id = str(sample.get("intent_id") or "").strip()
                idx = label_to_idx.get(intent_id)
                if idx is None:
                    continue
                item = dict(sample)
                item["label_idx"] = int(idx)
                filtered.append(item)

            if len(filtered) < max(30, len(runtime_intent_ids) * 3):
                raise RuntimeError(
                    f"insufficient labeled data for training: {len(filtered)} samples for {len(runtime_intent_ids)} intents"
                )

            texts = [str(row.get("text") or "") for row in filtered]
            labels = [int(row.get("label_idx")) for row in filtered]
            train_idx, val_idx = stratified_split(labels, val_ratio=float(val_ratio), random_seed=int(random_seed))
            if not train_idx:
                raise RuntimeError("stratified split produced empty train set")

            report_finetuned, artifact_finetuned = train_finetuned_model(
                model_name=self.model_name,
                device=self.device,
                finetuned_model_path=self.finetuned_model_path,
                finetuned_max_length=self.finetuned_max_length,
                finetuned_weight_decay=self.finetuned_weight_decay,
                texts=texts,
                labels=labels,
                intent_ids=runtime_intent_ids,
                train_idx=train_idx,
                val_idx=val_idx,
                random_seed=int(random_seed),
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
            )

            trained_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            version_id = f"tuned-{int(time.time())}"
            artifact = {
                "artifact_version": 4,
                "version_id": version_id,
                "trained_at": trained_at,
                "model_name": self.model_name,
                "intent_ids": runtime_intent_ids,
                "metrics": report_finetuned.get("metrics", {}),
                "dataset": {
                    **dataset_meta,
                    "samples_total": len(filtered),
                    "samples_train": len(train_idx),
                    "samples_val": len(val_idx),
                },
                "finetuned_model": artifact_finetuned,
            }
            self._save_artifact(output_path, artifact)
            self._activate_artifact(artifact)

            report = {
                "ok": True,
                "version_id": version_id,
                "trained_at": trained_at,
                "duration_sec": round(time.time() - started, 2),
                "output_path": output_path,
                "metrics": artifact["metrics"],
                "dataset": artifact["dataset"],
                "finetuned_model": report_finetuned,
            },
            self._set_last_train_report(report)
            return report
        except Exception as exc:
            self._set_last_train_error(str(exc))
            raise

    def _ensure_model_loaded(
        self,
        artifact: Dict[str, Any],
        model_intent_ids: List[str],
    ) -> Tuple[AutoModelForSequenceClassification, Any, int]:
        intent_key = tuple(model_intent_ids)
        finetuned_artifact = artifact.get("finetuned_model")
        if not isinstance(finetuned_artifact, dict):
            raise RuntimeError("finetuned model metadata is missing")
        model_path = str(finetuned_artifact.get("model_path") or "").strip()
        if not model_path:
            raise RuntimeError("finetuned model path is empty")

        with self._state_lock:
            if (
                self._model is not None
                and self._tokenizer is not None
                and self._active_intents == intent_key
                and self._active_model_path == model_path
            ):
                max_len = int(finetuned_artifact.get("max_length") or self.finetuned_max_length)
                return self._model, self._tokenizer, max_len

            model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if int(model.config.num_labels) != len(model_intent_ids):
                raise RuntimeError(
                    f"finetuned model classes mismatch: model={int(model.config.num_labels)} runtime={len(model_intent_ids)}"
                )

            self._model = model
            self._tokenizer = tokenizer
            self._active_intents = intent_key
            self._active_model_path = model_path
            max_len = int(finetuned_artifact.get("max_length") or self.finetuned_max_length)
            return model, tokenizer, max_len

    def _save_artifact(self, output_path: str, artifact: Dict[str, Any]) -> None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(artifact, tmp)
        os.replace(tmp, path)

    def _activate_artifact(self, artifact: Dict[str, Any]) -> None:
        with self._state_lock:
            self._artifact = dict(artifact)
            self._model = None
            self._tokenizer = None
            self._active_intents = None
            self._active_model_path = ""

    def _clear_artifact(self) -> None:
        with self._state_lock:
            self._artifact = None
            self._model = None
            self._tokenizer = None
            self._active_intents = None
            self._active_model_path = ""

    def _set_last_train_report(self, report: Dict[str, Any]) -> None:
        with self._state_lock:
            self._last_train_report = dict(report)
            self._last_train_error = ""

    def _set_last_train_error(self, error: str) -> None:
        with self._state_lock:
            self._last_train_error = str(error or "")

    def _artifact_intent_ids(self, artifact: Dict[str, Any]) -> List[str]:
        intent_ids = [str(x).strip() for x in list(artifact.get("intent_ids") or []) if str(x).strip()]
        if intent_ids:
            return intent_ids

        finetuned_model = artifact.get("finetuned_model")
        if isinstance(finetuned_model, dict):
            nested = [str(x).strip() for x in list(finetuned_model.get("intent_ids") or []) if str(x).strip()]
            if nested:
                return nested
        return []

    def _artifact_calibration(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        finetuned_model = artifact.get("finetuned_model")
        if isinstance(finetuned_model, dict) and isinstance(finetuned_model.get("calibration"), dict):
            return dict(finetuned_model.get("calibration") or {})
        if isinstance(artifact.get("calibration"), dict):
            return dict(artifact.get("calibration") or {})
        return {}

    def _artifact_temperature(self, artifact: Dict[str, Any]) -> float:
        calibration = self._artifact_calibration(artifact)
        try:
            temperature = float(calibration.get("temperature", 1.0))
        except Exception:
            temperature = 1.0
        if temperature <= 0.0:
            return 1.0
        return temperature

    def _same_intent_set(self, left: List[str], right: List[str]) -> bool:
        left_norm = self._comparable_intent_ids(left)
        right_norm = self._comparable_intent_ids(right)
        return len(left_norm) == len(right_norm) and set(left_norm) == set(right_norm)

    def _comparable_intent_ids(self, values: List[str]) -> List[str]:
        return [
            str(x).strip()
            for x in values
            if str(x).strip() and str(x).strip() != RESERVED_FALLBACK_INTENT_ID
        ]
