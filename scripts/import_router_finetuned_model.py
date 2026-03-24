#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import joblib
import torch

RESERVED_FALLBACK_INTENT_ID = "misc.triage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import external fine-tuned router model into local artifact format.")
    parser.add_argument("--source-model-dir", required=True, help="Path to Hugging Face model directory.")
    parser.add_argument(
        "--intents-path",
        default="configs/routing_intents.json",
        help="Path to routing intents JSON used by the router.",
    )
    parser.add_argument(
        "--target-model-dir",
        default="configs/router_finetuned_model",
        help="Directory where model files should be copied.",
    )
    parser.add_argument(
        "--artifact-path",
        default="configs/router_tuned_head.pt",
        help="Path to router tuned artifact (.pt).",
    )
    parser.add_argument(
        "--artifact-model-path",
        default="",
        help="Model path to store inside artifact. For Docker use /shared-config/router_finetuned_model.",
    )
    parser.add_argument(
        "--label-encoder-path",
        default="",
        help="Optional explicit path to label_encoder.joblib. Defaults to <source-model-dir>/label_encoder.joblib.",
    )
    parser.add_argument(
        "--temperature-file",
        default="",
        help="Optional explicit path to temperature_<target>.json produced by calibration step.",
    )
    parser.add_argument("--max-length", type=int, default=256, help="Max sequence length expected by the router.")
    parser.add_argument("--model-name", default="", help="Optional base model name for metadata.")
    parser.add_argument("--version-id", default="", help="Optional version id for the artifact.")
    parser.add_argument("--trained-at", default="", help="Optional trained_at timestamp in UTC ISO format.")
    parser.add_argument(
        "--exclude-runtime-intent",
        action="append",
        default=[],
        help="Runtime intent id to ignore during compatibility check. Can be repeated, e.g. spam.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print import plan without writing files.")
    return parser.parse_args()


def load_intents(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"intents payload must be an object: {path}")
    intent_ids = [str(x).strip() for x in payload.keys() if str(x).strip()]
    if not intent_ids:
        raise RuntimeError(f"no intents found in {path}")
    return sorted(intent_ids)


def normalize_intent_ids(values: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in values:
        value = str(item).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def comparable_intent_ids(values: Iterable[Any], excluded: Set[str] | None = None) -> List[str]:
    blocked = {RESERVED_FALLBACK_INTENT_ID}
    if excluded:
        blocked.update(str(value).strip() for value in excluded if str(value).strip())
    return [value for value in normalize_intent_ids(values) if value not in blocked]


def extract_intent_ids_from_label_encoder(obj: Any) -> List[str]:
    if hasattr(obj, "classes_"):
        return normalize_intent_ids(getattr(obj, "classes_"))
    if isinstance(obj, dict):
        if "classes_" in obj:
            return normalize_intent_ids(obj.get("classes_") or [])
        if "intent_ids" in obj:
            return normalize_intent_ids(obj.get("intent_ids") or [])
    if isinstance(obj, (list, tuple)):
        return normalize_intent_ids(obj)
    return []


def load_model_intent_ids(source_dir: Path, explicit_label_encoder: Path | None) -> Tuple[List[str], str]:
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
            intent_ids = extract_intent_ids_from_label_encoder(obj)
            if intent_ids:
                return intent_ids, source_name
        elif path.name == "intent_ids.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                intent_ids = normalize_intent_ids(payload.get("intent_ids") or [])
            elif isinstance(payload, list):
                intent_ids = normalize_intent_ids(payload)
            else:
                intent_ids = []
            if intent_ids:
                return intent_ids, source_name
        elif path.name == "config.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            id2label = payload.get("id2label")
            if isinstance(id2label, dict):
                ordered = [label for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
                intent_ids = normalize_intent_ids(ordered)
                if intent_ids and not all(value.upper().startswith("LABEL_") for value in intent_ids):
                    return intent_ids, source_name

    raise RuntimeError(
        "failed to determine intent order from external model; provide label_encoder.joblib or intent_ids.json"
    )


def ensure_source_model_complete(path: Path) -> None:
    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise RuntimeError(f"source model dir is missing required files: {', '.join(missing)}")
    if not ((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists()):
        raise RuntimeError("source model dir must contain model.safetensors or pytorch_model.bin")


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
        raise RuntimeError(
            f"multiple temperature files found in source model dir, use --temperature-file explicitly: {names}"
        )

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
    if source_dir.resolve() == target_dir.resolve():
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_dir.iterdir():
        dest = target_dir / item.name
        if item.resolve() == dest.resolve():
            continue
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)


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
    intents_path = Path(args.intents_path).expanduser().resolve()
    target_model_dir = Path(args.target_model_dir).expanduser().resolve()
    artifact_path = Path(args.artifact_path).expanduser().resolve()
    label_encoder_path = None
    temperature_path = None
    if str(args.label_encoder_path).strip():
        label_encoder_path = Path(args.label_encoder_path).expanduser().resolve()
    if str(args.temperature_file).strip():
        temperature_path = Path(args.temperature_file).expanduser().resolve()
    excluded_runtime_intents = {str(value).strip() for value in list(args.exclude_runtime_intent or []) if str(value).strip()}

    if not source_dir.exists() or not source_dir.is_dir():
        raise RuntimeError(f"source model dir not found: {source_dir}")
    if not intents_path.exists() or not intents_path.is_file():
        raise RuntimeError(f"intents file not found: {intents_path}")
    if label_encoder_path is not None and not label_encoder_path.exists():
        raise RuntimeError(f"label encoder not found: {label_encoder_path}")
    if temperature_path is not None and not temperature_path.exists():
        raise RuntimeError(f"temperature file not found: {temperature_path}")

    ensure_source_model_complete(source_dir)
    runtime_intents = load_intents(intents_path)
    model_intents, mapping_source = load_model_intent_ids(source_dir, label_encoder_path)
    calibration = load_temperature_calibration(source_dir, temperature_path)

    runtime_set = set(comparable_intent_ids(runtime_intents, excluded=excluded_runtime_intents))
    model_set = set(comparable_intent_ids(model_intents, excluded=excluded_runtime_intents))
    missing_in_model = sorted(runtime_set - model_set)
    extra_in_model = sorted(model_set - runtime_set)
    if missing_in_model or extra_in_model:
        raise RuntimeError(
            "model intents do not match router intents; "
            f"missing_in_model={missing_in_model}, extra_in_model={extra_in_model}"
        )

    trained_at = str(args.trained_at).strip() or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    version_id = str(args.version_id).strip() or f"imported-{int(time.time())}"
    model_name = detect_model_name(source_dir, args.model_name)
    artifact_model_path = str(args.artifact_model_path).strip() or str(target_model_dir)

    meta_payload = {
        "model_name": model_name,
        "intent_ids": model_intents,
        "trained_at": trained_at,
        "max_length": int(args.max_length),
        "import_source": str(source_dir),
        "label_mapping_source": mapping_source,
    }
    if calibration:
        meta_payload["calibration"] = calibration
    artifact = {
        "artifact_version": 4,
        "version_id": version_id,
        "trained_at": trained_at,
        "model_name": model_name,
        "intent_ids": model_intents,
        "metrics": {},
        "dataset": {
            "import_source": str(source_dir),
            "label_mapping_source": mapping_source,
        },
        "calibration": calibration,
        "finetuned_model": {
            "enabled": True,
            "model_path": artifact_model_path,
            "intent_ids": model_intents,
            "trained_at": trained_at,
            "max_length": int(args.max_length),
            "metrics": {},
            "dataset": {
                "import_source": str(source_dir),
                "label_mapping_source": mapping_source,
            },
            "calibration": calibration,
        },
    }

    print(f"source_model_dir={source_dir}")
    print(f"target_model_dir={target_model_dir}")
    print(f"artifact_path={artifact_path}")
    print(f"artifact_model_path={artifact_model_path}")
    print(f"intents_path={intents_path}")
    print(f"intent_count={len(model_intents)}")
    print(f"excluded_runtime_intents={sorted(excluded_runtime_intents)}")
    print(f"label_mapping_source={mapping_source}")
    print(f"temperature={calibration.get('temperature', 1.0)}")
    print(f"version_id={version_id}")

    if args.dry_run:
        return 0

    copy_model_tree(source_dir, target_model_dir)
    (target_model_dir / "intent_ids.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, artifact_path)

    print("import_status=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
