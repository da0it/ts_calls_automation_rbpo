#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import torch


DEFAULT_POSITIVE_LABEL = "spam"
DEFAULT_BACKEND = "sklearn_tfidf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import external binary spam classifier into router spam-gate format.")
    parser.add_argument("--source-model-dir", required=True, help="Path to Hugging Face model directory.")
    parser.add_argument(
        "--backend",
        default=DEFAULT_BACKEND,
        choices=["hf_sequence_classifier", "sklearn_tfidf"],
        help="Runtime backend for the spam gate artifact.",
    )
    parser.add_argument("--target-model-dir", default="configs/router_spam_model", help="Directory where model files should be copied.")
    parser.add_argument("--artifact-path", default="configs/router_spam_gate.pt", help="Path to spam gate artifact (.pt).")
    parser.add_argument("--artifact-model-path", default="", help="Model path to store inside artifact. For Docker use /shared-config/router_spam_model.")
    parser.add_argument("--label-encoder-path", default="", help="Optional explicit path to label_encoder.joblib.")
    parser.add_argument("--temperature-file", default="", help="Optional explicit path to temperature_*.json.")
    parser.add_argument("--classifier-path", default="", help="Optional explicit path to sklearn classifier joblib.")
    parser.add_argument("--vectorizer-path", default="", help="Optional explicit path to TF-IDF vectorizer joblib.")
    parser.add_argument("--positive-label", default=DEFAULT_POSITIVE_LABEL, help="Positive class label meaning spam.")
    parser.add_argument("--threshold", type=float, default=0.8, help="Spam confidence threshold for stage-1 gate.")
    parser.add_argument("--allow-threshold", type=float, default=0.35, help="Low spam probability threshold for automatic non-spam pass.")
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length expected by the router.")
    parser.add_argument("--model-name", default="", help="Optional base model name for metadata.")
    parser.add_argument("--version-id", default="", help="Optional version id for the artifact.")
    parser.add_argument("--trained-at", default="", help="Optional trained_at timestamp in UTC ISO format.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print import plan without writing files.")
    return parser.parse_args()


def normalize_labels(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def extract_labels_from_label_encoder(obj: Any) -> List[str]:
    if hasattr(obj, "classes_"):
        return normalize_labels(getattr(obj, "classes_"))
    if isinstance(obj, dict):
        if "classes_" in obj:
            return normalize_labels(obj.get("classes_") or [])
        if "intent_ids" in obj:
            return normalize_labels(obj.get("intent_ids") or [])
    if isinstance(obj, (list, tuple)):
        return normalize_labels(obj)
    return []


def load_model_labels(
    source_dir: Path,
    explicit_label_encoder: Path | None,
    classifier_path: Path | None = None,
) -> Tuple[List[str], str]:
    candidates: List[Tuple[Path, str]] = []
    if explicit_label_encoder is not None:
        candidates.append((explicit_label_encoder, "label_encoder.joblib"))
    else:
        candidates.append((source_dir / "label_encoder.joblib", "label_encoder.joblib"))
        candidates.append((source_dir / "intent_ids.json", "intent_ids.json"))
        candidates.append((source_dir / "config.json", "config.json:id2label"))

    for path, source_name in candidates:
        if not path.exists():
            continue
        if path.name == "label_encoder.joblib":
            obj = joblib.load(path)
            labels = extract_labels_from_label_encoder(obj)
            if labels:
                return labels, source_name
        elif path.name == "intent_ids.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                labels = normalize_labels(payload.get("intent_ids") or [])
            elif isinstance(payload, list):
                labels = normalize_labels(payload)
            else:
                labels = []
            if labels:
                return labels, source_name
        elif path.name == "config.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            id2label = payload.get("id2label")
            if isinstance(id2label, dict):
                ordered = [label for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
                labels = normalize_labels(ordered)
                if labels and not all(value.upper().startswith("LABEL_") for value in labels):
                    return labels, source_name

    if classifier_path is not None and classifier_path.exists():
        model = joblib.load(classifier_path)
        labels = extract_labels_from_label_encoder(model)
        if labels:
            return labels, "classifier.joblib:classes_"

    raise RuntimeError("failed to determine label order from external model; provide label_encoder.joblib or intent_ids.json")


def ensure_source_model_complete(path: Path) -> None:
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"source model dir is missing required files: {', '.join(missing)}")
    if not ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()):
        raise RuntimeError("source model dir must contain model.safetensors or pytorch_model.bin")


def resolve_optional_joblib(source_dir: Path, explicit_path: Path | None, names: List[str], label: str) -> Path:
    if explicit_path is not None:
        if not explicit_path.exists():
            raise RuntimeError(f"{label} not found: {explicit_path}")
        return explicit_path
    for name in names:
        path = source_dir / name
        if path.exists():
            return path
    raise RuntimeError(f"failed to locate {label}; checked: {', '.join(names)}")


def load_temperature_calibration(source_dir: Path, explicit_temperature_path: Path | None) -> Dict[str, Any]:
    candidates: List[Path] = []
    if explicit_temperature_path is not None:
        candidates.append(explicit_temperature_path)
    else:
        candidates.extend(sorted(source_dir.glob("temperature_*.json")))
    if not candidates:
        return {}
    if explicit_temperature_path is None and len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise RuntimeError(f"multiple temperature files found in source model dir, use --temperature-file explicitly: {names}")

    path = candidates[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"temperature file must be a JSON object: {path}")
    try:
        temperature = float(payload.get("temperature"))
    except Exception as exc:
        raise RuntimeError(f"temperature field is missing or invalid in {path}") from exc
    if temperature <= 0.0:
        raise RuntimeError(f"temperature must be > 0 in {path}")
    calibration = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "source_file": path.name,
    }
    target = str(payload.get("target") or "").strip()
    if target:
        calibration["target"] = target
    return calibration


def copy_model_tree(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


def copy_if_external(path: Path | None, source_dir: Path, target_dir: Path) -> None:
    if path is None:
        return
    try:
        path.relative_to(source_dir)
        return
    except ValueError:
        pass
    shutil.copy2(path, target_dir / path.name)


def detect_model_name(source_dir: Path, override: str) -> str:
    if override.strip():
        return override.strip()
    config_path = source_dir / "config.json"
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        raw_name = str(payload.get("_name_or_path") or "").strip()
        if raw_name:
            return raw_name
    return source_dir.name


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_model_dir).expanduser().resolve()
    target_model_dir = Path(args.target_model_dir).expanduser().resolve()
    artifact_path = Path(args.artifact_path).expanduser().resolve()
    label_encoder_path = Path(args.label_encoder_path).expanduser().resolve() if str(args.label_encoder_path).strip() else None
    temperature_path = Path(args.temperature_file).expanduser().resolve() if str(args.temperature_file).strip() else None
    classifier_path = Path(args.classifier_path).expanduser().resolve() if str(args.classifier_path).strip() else None
    vectorizer_path = Path(args.vectorizer_path).expanduser().resolve() if str(args.vectorizer_path).strip() else None

    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(f"source model dir not found: {source_dir}")
    if label_encoder_path is not None and not label_encoder_path.exists():
        raise RuntimeError(f"label encoder not found: {label_encoder_path}")
    if temperature_path is not None and not temperature_path.exists():
        raise RuntimeError(f"temperature file not found: {temperature_path}")

    backend = str(args.backend).strip() or DEFAULT_BACKEND
    calibration: Dict[str, Any] = {}
    classifier_file_name = ""
    vectorizer_file_name = ""
    resolved_classifier_path: Path | None = None
    resolved_vectorizer_path: Path | None = None

    if backend == "hf_sequence_classifier":
        ensure_source_model_complete(source_dir)
        calibration = load_temperature_calibration(source_dir, temperature_path)
    else:
        resolved_classifier_path = resolve_optional_joblib(
            source_dir,
            classifier_path,
            ["model.joblib", "classifier.joblib", "svm.joblib", "svm_model.joblib"],
            "classifier joblib",
        )
        resolved_vectorizer_path = resolve_optional_joblib(
            source_dir,
            vectorizer_path,
            ["vectorizer.joblib", "tfidf.joblib", "tfidf_vectorizer.joblib"],
            "vectorizer joblib",
        )
        classifier_file_name = resolved_classifier_path.name
        vectorizer_file_name = resolved_vectorizer_path.name

    labels, mapping_source = load_model_labels(source_dir, label_encoder_path, resolved_classifier_path)

    positive_label = str(args.positive_label).strip()
    if not positive_label:
        raise RuntimeError("positive label is empty")
    if positive_label not in labels:
        raise RuntimeError(f"positive label {positive_label!r} not found in model labels: {labels}")
    if len(labels) != 2:
        raise RuntimeError(f"spam gate expects binary classifier with 2 labels, got {len(labels)}: {labels}")

    trained_at = str(args.trained_at).strip() or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    version_id = str(args.version_id).strip() or f"spam-gate-{int(time.time())}"
    model_name = detect_model_name(source_dir, args.model_name)
    artifact_model_path = str(args.artifact_model_path).strip() or str(target_model_dir)
    threshold = float(max(0.0, min(1.0, args.threshold)))
    allow_threshold = float(max(0.0, min(threshold, args.allow_threshold)))

    gate_meta = {
        "enabled": True,
        "backend": backend,
        "model_path": artifact_model_path,
        "labels": labels,
        "positive_label": positive_label,
        "trained_at": trained_at,
        "max_length": int(args.max_length),
        "threshold": threshold,
        "allow_threshold": allow_threshold,
        "calibration": calibration,
    }
    if classifier_file_name:
        gate_meta["classifier_file"] = classifier_file_name
    if vectorizer_file_name:
        gate_meta["vectorizer_file"] = vectorizer_file_name
    artifact = {
        "artifact_version": 1,
        "gate_type": "spam_gate",
        "backend": backend,
        "version_id": version_id,
        "trained_at": trained_at,
        "model_name": model_name,
        "labels": labels,
        "label_mapping_source": mapping_source,
        "spam_gate": gate_meta,
    }

    print(f"source_model_dir={source_dir}")
    print(f"target_model_dir={target_model_dir}")
    print(f"artifact_path={artifact_path}")
    print(f"artifact_model_path={artifact_model_path}")
    print(f"backend={backend}")
    print(f"labels={labels}")
    print(f"positive_label={positive_label}")
    print(f"threshold={threshold}")
    print(f"allow_threshold={allow_threshold}")
    print(f"temperature={calibration.get('temperature', 1.0)}")
    print(f"version_id={version_id}")

    if args.dry_run:
        return 0

    copy_model_tree(source_dir, target_model_dir)
    copy_if_external(resolved_classifier_path, source_dir, target_model_dir)
    copy_if_external(resolved_vectorizer_path, source_dir, target_model_dir)
    (target_model_dir / "spam_gate_meta.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "labels": labels,
                "positive_label": positive_label,
                "threshold": threshold,
                "allow_threshold": allow_threshold,
                "backend": backend,
                "trained_at": trained_at,
                "max_length": int(args.max_length),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)
    print("import_status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
