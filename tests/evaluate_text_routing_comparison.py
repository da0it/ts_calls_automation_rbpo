#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import joblib
except ModuleNotFoundError:  # pragma: no cover - handled at runtime with a clearer message.
    joblib = None  # type: ignore[assignment]

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
except ModuleNotFoundError:  # pragma: no cover
    accuracy_score = classification_report = confusion_matrix = f1_score = None  # type: ignore[assignment]

try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError:  # pragma: no cover
    AutoModelForSequenceClassification = AutoTokenizer = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[1]
ROUTER_ROOT = REPO_ROOT / "services" / "router"
if str(ROUTER_ROOT) not in sys.path:
    sys.path.insert(0, str(ROUTER_ROOT))

SPAM_RUNTIME_INTENT = "spam.call"
TRIAGE_INTENT = "misc.triage"
SPAM_LABEL = "spam"


def require_ml_dependencies() -> None:
    missing = []
    if joblib is None:
        missing.append("joblib")
    if torch is None:
        missing.append("torch")
    if accuracy_score is None or classification_report is None or confusion_matrix is None or f1_score is None:
        missing.append("scikit-learn")
    if AutoModelForSequenceClassification is None or AutoTokenizer is None:
        missing.append("transformers")
    if missing:
        raise RuntimeError(
            "missing Python dependencies: "
            + ", ".join(missing)
            + ". Activate the project venv or install router requirements first."
        )


@dataclass
class Prediction:
    label: str
    confidence: float
    raw_label: str
    details: Dict[str, Any]


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t;,")
        return dialect.delimiter
    except Exception:
        counts = {"\t": sample.count("\t"), ";": sample.count(";"), ",": sample.count(",")}
        return max(counts.items(), key=lambda item: item[1])[0]


def read_rows(path: Path) -> Tuple[List[Dict[str, str]], List[str], str]:
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers, delimiter


def clean(value: Any) -> str:
    return str(value or "").strip()


def normalize_label(value: Any) -> str:
    raw = clean(value).lower()
    if raw in {"", "none", "null", "nan", "-"}:
        return ""
    aliases = {
        "spam": SPAM_LABEL,
        "спам": SPAM_LABEL,
        "spam.call": SPAM_LABEL,
        "consultation.general": "consulting",
        "training.courses": "courses",
        "portal access": "portal_access",
        "portal-access": "portal_access",
        "portal.access": "portal_access",
    }
    if raw in aliases:
        return aliases[raw]
    return raw.replace(" ", "_").replace("-", "_")


def is_spam_value(value: Any) -> Optional[bool]:
    raw = clean(value).lower()
    if raw in {"spam", "спам", "1", "true", "yes", "y", "да"}:
        return True
    if raw in {"not_spam", "non_spam", "ham", "0", "false", "no", "n", "нет"}:
        return False
    return None


def gold_label(row: Dict[str, str]) -> str:
    manual_spam = is_spam_value(row.get("manual_is_spam"))
    if manual_spam is True:
        return SPAM_LABEL
    purpose = normalize_label(row.get("call_purpose"))
    if purpose:
        return purpose
    if manual_spam is False:
        return ""
    return normalize_label(row.get("manual_is_spam"))


def is_synthetic(row: Dict[str, str]) -> bool:
    raw = clean(row.get("is_synthetic")).lower()
    return raw in {"1", "true", "yes", "y", "да"}


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_labels_from_encoder(obj: Any) -> List[str]:
    if hasattr(obj, "classes_"):
        return [normalize_label(x) for x in getattr(obj, "classes_")]
    if isinstance(obj, dict):
        if "classes_" in obj:
            return [normalize_label(x) for x in obj.get("classes_") or []]
        if "intent_ids" in obj:
            return [normalize_label(x) for x in obj.get("intent_ids") or []]
    if isinstance(obj, (list, tuple)):
        return [normalize_label(x) for x in obj]
    return []


def load_model_labels(model_dir: Path, label_encoder_path: Optional[Path]) -> List[str]:
    candidates: List[Path] = []
    if label_encoder_path is not None:
        candidates.append(label_encoder_path)
    candidates.extend([model_dir / "label_encoder.joblib", model_dir / "intent_ids.json", model_dir / "config.json"])

    for path in candidates:
        if not path.exists():
            continue
        if path.name == "label_encoder.joblib":
            labels = extract_labels_from_encoder(joblib.load(path))
            if labels:
                return labels
        if path.name == "intent_ids.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("intent_ids") if isinstance(payload, dict) else payload
            labels = [normalize_label(x) for x in values or []]
            if labels:
                return labels
        if path.name == "config.json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            id2label = payload.get("id2label")
            if isinstance(id2label, dict):
                ordered = [label for _, label in sorted(id2label.items(), key=lambda kv: int(kv[0]))]
                labels = [normalize_label(x) for x in ordered]
                if labels and not all(label.startswith("label_") for label in labels):
                    return labels

    raise RuntimeError(f"failed to determine labels for model: {model_dir}")


class HFTextClassifier:
    def __init__(
        self,
        *,
        model_dir: Path,
        label_encoder_path: Optional[Path],
        device: str,
        max_length: int,
    ) -> None:
        self.model_dir = model_dir
        self.labels = load_model_labels(model_dir, label_encoder_path)
        self.device = device
        self.max_length = int(max(64, min(512, max_length)))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
        self.model.eval()
        if int(self.model.config.num_labels) != len(self.labels):
            raise RuntimeError(
                f"label count mismatch for {model_dir}: model={self.model.config.num_labels}, labels={len(self.labels)}"
            )

    def predict(self, text: str) -> Prediction:
        enc = self.tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {key: value.to(self.device) for key, value in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=1).squeeze(0)
        best_idx = int(torch.argmax(probs).item())
        top_k = min(3, len(self.labels))
        top_indices = torch.topk(probs, k=top_k).indices.tolist()
        top3 = [{"label": self.labels[int(idx)], "score": float(probs[int(idx)].item())} for idx in top_indices]
        label = self.labels[best_idx]
        return Prediction(
            label=label,
            raw_label=label,
            confidence=float(probs[best_idx].item()),
            details={"top3": top3},
        )


def load_torch_artifact(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid torch artifact: {path}")
    return payload


def patch_spam_gate_model_path(analyzer: RubertEmbeddingAnalyzer, spam_model_dir: Optional[Path]) -> None:
    if spam_model_dir is None:
        return
    spam_gate = getattr(analyzer, "_spam_gate", None)
    if spam_gate is None:
        return
    artifact = getattr(spam_gate, "_artifact", None)
    if not isinstance(artifact, dict):
        return
    gate_meta = artifact.get("spam_gate")
    if not isinstance(gate_meta, dict):
        return
    gate_meta["model_path"] = str(spam_model_dir.resolve())
    setattr(spam_gate, "_artifact", artifact)
    setattr(spam_gate, "_model", None)
    setattr(spam_gate, "_vectorizer", None)
    setattr(spam_gate, "_active_model_path", "")


def build_two_stage_analyzer(args: argparse.Namespace) -> Any:
    from routing.ai_analyzer import RubertEmbeddingAnalyzer

    analyzer = RubertEmbeddingAnalyzer(
        model_name=args.stage2_model_name,
        device=args.device,
        min_confidence=args.router_min_confidence,
        max_text_chars=args.max_text_chars,
        tuned_model_path=str(args.stage2_artifact_path),
        finetuned_enabled=True,
        finetuned_model_path=str(args.stage2_model_dir),
        finetuned_max_length=args.stage2_max_length,
        nlp_backend=args.nlp_backend,
        nlp_text_mode=args.nlp_text_mode,
        nlp_stanza_resources_dir=args.nlp_stanza_resources_dir,
        spam_gate_enabled=True,
        spam_gate_model_path=str(args.spam_model_dir),
        spam_gate_artifact_path=str(args.spam_artifact_path),
        spam_gate_threshold=args.spam_gate_threshold,
        spam_gate_allow_threshold=args.spam_gate_allow_threshold,
        spam_gate_score_threshold=args.spam_gate_score_threshold,
        spam_gate_score_allow_threshold=args.spam_gate_score_allow_threshold,
        spam_gate_positive_label=args.spam_gate_positive_label,
        spam_conflict_review_min_confidence=args.spam_conflict_review_min_confidence,
    )
    patch_spam_gate_model_path(analyzer, args.spam_model_dir)
    return analyzer


def predict_two_stage(
    analyzer: Any,
    intents: Dict[str, Dict[str, Any]],
    *,
    row_id: int,
    text: str,
    auto_threshold: float,
) -> Prediction:
    from routing.models import CallInput, Segment

    call = CallInput(
        call_id=f"text-row-{row_id}",
        segments=[Segment(start=0.0, end=0.0, speaker="speaker_0", role="client", text=text)],
    )
    analysis = analyzer.analyze(call, intents)
    raw_intent = clean(analysis.intent.intent_id)
    label = normalize_label(raw_intent)
    spam_decision = analysis.raw.get("spam_decision") if isinstance(analysis.raw, dict) else {}
    if not isinstance(spam_decision, dict):
        spam_decision = {}
    confidence = float(analysis.intent.confidence or 0.0)
    mode = clean(analysis.raw.get("mode")) if isinstance(analysis.raw, dict) else ""
    review_required = (
        mode in {"spam_gate_review", "finetuned_low_confidence_review", "finetuned_spam_conflict_review"}
        or clean(spam_decision.get("status")) == "review"
        or (label != SPAM_LABEL and confidence < auto_threshold)
    )
    return Prediction(
        label=label,
        raw_label=raw_intent,
        confidence=confidence,
        details={
            "mode": mode,
            "spam_status": clean(spam_decision.get("status")),
            "spam_reason": clean(spam_decision.get("reason")),
            "spam_confidence": spam_decision.get("confidence"),
            "review_required": review_required,
            "auto_ticket_allowed": bool(label != SPAM_LABEL and confidence >= auto_threshold and not review_required),
            "raw": analysis.raw,
        },
    )


def prepare_router_text(text: str, args: argparse.Namespace) -> str:
    from routing.nlp_preprocess import PreprocessConfig, build_canonical

    prep = build_canonical(
        [(0.0, text, "client")],
        PreprocessConfig(
            backend=args.nlp_backend,
            model_text_mode=args.nlp_text_mode,
            drop_fillers=True,
            dedupe=True,
            keep_timestamps=True,
            do_lemmatize=True,
            drop_stopwords=False,
            max_chars=args.max_text_chars,
            stanza_resources_dir=args.nlp_stanza_resources_dir,
        ),
    )
    return prep.model_text


def safe_report(y_true: List[str], y_pred: List[str], labels: Optional[List[str]] = None) -> Dict[str, Any]:
    if not y_true:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "classification_report": {},
            "confusion_matrix": {"labels": [], "matrix": []},
        }
    report_labels = labels or sorted(set(y_true) | set(y_pred))
    return {
        "samples": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=report_labels, average="macro", zero_division=0)), 6),
        "weighted_f1": round(float(f1_score(y_true, y_pred, labels=report_labels, average="weighted", zero_division=0)), 6),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=report_labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": {
            "labels": report_labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=report_labels).tolist(),
        },
    }


def binary_labels(values: Iterable[str]) -> List[str]:
    return [SPAM_LABEL if value == SPAM_LABEL else "non_spam" for value in values]


def build_metrics(records: List[Dict[str, Any]], pred_col: str) -> Dict[str, Any]:
    valid = [row for row in records if row.get("gold_label") and row.get(pred_col)]
    y_true = [row["gold_label"] for row in valid]
    y_pred = [row[pred_col] for row in valid]
    all_labels = sorted(set(y_true) | set(y_pred))
    nonspam = [row for row in valid if row["gold_label"] != SPAM_LABEL]
    y_true_nonspam = [row["gold_label"] for row in nonspam]
    y_pred_nonspam = [row[pred_col] for row in nonspam]
    nonspam_labels = sorted(set(y_true_nonspam) | set(y_pred_nonspam))

    return {
        "multiclass": safe_report(y_true, y_pred, labels=all_labels),
        "binary_spam": safe_report(binary_labels(y_true), binary_labels(y_pred), labels=[SPAM_LABEL, "non_spam"]),
        "nonspam_intents": safe_report(y_true_nonspam, y_pred_nonspam, labels=nonspam_labels),
    }


def split_metrics(records: List[Dict[str, Any]], pred_col: str) -> Dict[str, Any]:
    out = {"all": build_metrics(records, pred_col)}
    if any("is_synthetic" in row for row in records):
        real = [row for row in records if not row.get("is_synthetic")]
        synthetic = [row for row in records if row.get("is_synthetic")]
        out["real_only"] = build_metrics(real, pred_col)
        out["synthetic_only"] = build_metrics(synthetic, pred_col)
    return out


def format_report_block(title: str, metrics: Dict[str, Any]) -> str:
    multi = metrics["multiclass"]
    binary = metrics["binary_spam"]
    nonspam = metrics["nonspam_intents"]
    return "\n".join(
        [
            f"== {title} ==",
            f"multiclass: samples={multi['samples']} accuracy={multi['accuracy']:.4f} "
            f"macro_f1={multi['macro_f1']:.4f} weighted_f1={multi['weighted_f1']:.4f}",
            f"binary spam: samples={binary['samples']} accuracy={binary['accuracy']:.4f} "
            f"macro_f1={binary['macro_f1']:.4f} weighted_f1={binary['weighted_f1']:.4f}",
            f"nonspam intents: samples={nonspam['samples']} accuracy={nonspam['accuracy']:.4f} "
            f"macro_f1={nonspam['macro_f1']:.4f} weighted_f1={nonspam['weighted_f1']:.4f}",
        ]
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def parse_optional_float(value: str) -> Optional[float]:
    raw = clean(value)
    if not raw:
        return None
    return float(raw)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a single 6-class text classifier with the two-stage router pipeline on a text CSV/TSV dataset."
    )
    parser.add_argument("--csv", required=True, help="CSV/TSV with text, manual_is_spam, call_purpose, is_synthetic.")
    parser.add_argument("--single-model-dir", required=True, help="HF directory of the 6-class model.")
    parser.add_argument("--single-label-encoder-path", default="", help="Optional label_encoder.joblib for the 6-class model.")
    parser.add_argument("--single-max-length", type=int, default=512)
    parser.add_argument("--single-raw-text", action="store_true", help="Use raw dataset text for the 6-class model.")

    parser.add_argument("--intents-path", default="configs/routing_intents.json")
    parser.add_argument("--stage2-artifact-path", default="configs/router_tuned_head.pt")
    parser.add_argument("--stage2-model-dir", default="configs/router_finetuned_model")
    parser.add_argument("--stage2-model-name", default="xlm-roberta-large")
    parser.add_argument("--stage2-max-length", type=int, default=512)
    parser.add_argument("--spam-artifact-path", default="configs/router_spam_gate.pt")
    parser.add_argument("--spam-model-dir", default="configs/router_spam_model")
    parser.add_argument("--spam-gate-threshold", type=float, default=0.85)
    parser.add_argument("--spam-gate-allow-threshold", type=float, default=0.60)
    parser.add_argument("--spam-gate-score-threshold", type=parse_optional_float, default=None)
    parser.add_argument("--spam-gate-score-allow-threshold", type=parse_optional_float, default=None)
    parser.add_argument("--spam-gate-positive-label", default="spam")
    parser.add_argument("--spam-conflict-review-min-confidence", type=float, default=0.98)

    parser.add_argument("--router-min-confidence", type=float, default=0.50)
    parser.add_argument("--auto-threshold", type=float, default=0.80)
    parser.add_argument("--max-text-chars", type=int, default=4000)
    parser.add_argument("--nlp-backend", default="stanza", choices=["stanza", "none"])
    parser.add_argument("--nlp-text-mode", default="canonical", choices=["canonical", "plain", "normalized", "tokens", "lemmas"])
    parser.add_argument("--nlp-stanza-resources-dir", default="")

    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--exclude-synthetic", action="store_true")
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-text", default="")
    args = parser.parse_args()

    args.csv = Path(args.csv).expanduser().resolve()
    args.single_model_dir = Path(args.single_model_dir).expanduser().resolve()
    args.single_label_encoder_path = (
        Path(args.single_label_encoder_path).expanduser().resolve() if args.single_label_encoder_path else None
    )
    args.intents_path = Path(args.intents_path).expanduser().resolve()
    args.stage2_artifact_path = Path(args.stage2_artifact_path).expanduser().resolve()
    args.stage2_model_dir = Path(args.stage2_model_dir).expanduser().resolve()
    if not args.stage2_artifact_path.exists():
        fallback = args.stage2_model_dir / "router_tuned_head.pt"
        if fallback.exists():
            args.stage2_artifact_path = fallback.resolve()
    args.spam_artifact_path = Path(args.spam_artifact_path).expanduser().resolve()
    args.spam_model_dir = Path(args.spam_model_dir).expanduser().resolve()
    if args.device == "auto":
        if torch is not None and torch.cuda.is_available():
            args.device = "cuda"
        elif torch is not None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    return args


def validate_paths(args: argparse.Namespace) -> None:
    required = [
        ("csv", args.csv),
        ("single model dir", args.single_model_dir),
        ("intents", args.intents_path),
        ("stage2 artifact", args.stage2_artifact_path),
        ("stage2 model dir", args.stage2_model_dir),
        ("spam artifact", args.spam_artifact_path),
        ("spam model dir", args.spam_model_dir),
    ]
    missing = [f"{label}: {path}" for label, path in required if not path.exists()]
    if missing:
        raise RuntimeError("missing required path(s):\n" + "\n".join(missing))


def main() -> int:
    args = parse_args()
    require_ml_dependencies()
    validate_paths(args)

    rows, headers, delimiter = read_rows(args.csv)
    if not rows:
        print(f"[ERROR] dataset is empty: {args.csv}")
        return 2
    for required_col in ["text", "manual_is_spam", "call_purpose"]:
        if required_col not in headers:
            print(f"[ERROR] missing required column: {required_col}")
            return 2

    if args.exclude_synthetic:
        rows = [row for row in rows if not is_synthetic(row)]
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"Dataset: {args.csv}")
    print(f"Rows: {len(rows)}")
    print(f"Delimiter: {repr(delimiter)}")
    print(f"Device: {args.device}")
    print(f"Single 6-class model: {args.single_model_dir}")
    print(f"Two-stage spam model: {args.spam_model_dir}")
    print(f"Two-stage stage2 model: {args.stage2_model_dir}")

    intents = load_json(args.intents_path)
    single_model = HFTextClassifier(
        model_dir=args.single_model_dir,
        label_encoder_path=args.single_label_encoder_path,
        device=args.device,
        max_length=args.single_max_length,
    )
    two_stage = build_two_stage_analyzer(args)

    records: List[Dict[str, Any]] = []
    started = time.time()

    for idx, row in enumerate(rows, start=1):
        text = clean(row.get("text"))
        gold = gold_label(row)
        record: Dict[str, Any] = {
            **row,
            "row_id": idx,
            "gold_label": gold,
            "is_synthetic": is_synthetic(row),
            "single6_pred": "",
            "single6_confidence": "",
            "two_stage_pred": "",
            "two_stage_raw_intent": "",
            "two_stage_confidence": "",
            "two_stage_mode": "",
            "two_stage_spam_status": "",
            "two_stage_spam_reason": "",
            "two_stage_spam_confidence": "",
            "two_stage_review_required": "",
            "two_stage_auto_ticket_allowed": "",
            "error": "",
        }
        try:
            if not text:
                raise RuntimeError("empty text")
            single_input = text if args.single_raw_text else prepare_router_text(text, args)
            single = single_model.predict(single_input)
            two = predict_two_stage(two_stage, intents, row_id=idx, text=text, auto_threshold=args.auto_threshold)

            record.update(
                {
                    "single6_pred": single.label,
                    "single6_confidence": round(single.confidence, 6),
                    "single6_top3": json.dumps(single.details.get("top3", []), ensure_ascii=False),
                    "two_stage_pred": two.label,
                    "two_stage_raw_intent": two.raw_label,
                    "two_stage_confidence": round(two.confidence, 6),
                    "two_stage_mode": two.details.get("mode", ""),
                    "two_stage_spam_status": two.details.get("spam_status", ""),
                    "two_stage_spam_reason": two.details.get("spam_reason", ""),
                    "two_stage_spam_confidence": two.details.get("spam_confidence", ""),
                    "two_stage_review_required": int(bool(two.details.get("review_required"))),
                    "two_stage_auto_ticket_allowed": int(bool(two.details.get("auto_ticket_allowed"))),
                }
            )
            if idx == 1 or idx % 25 == 0 or idx == len(rows):
                print(f"[{idx}/{len(rows)}] gold={gold} single6={single.label} two_stage={two.label}")
        except Exception as exc:
            record["error"] = str(exc)
            print(f"[FAIL] row {idx}: {exc}")
        records.append(record)

    ok_records = [row for row in records if not row.get("error")]
    single_metrics = split_metrics(ok_records, "single6_pred")
    two_stage_metrics = split_metrics(ok_records, "two_stage_pred")
    auto_allowed = sum(1 for row in ok_records if str(row.get("two_stage_auto_ticket_allowed")) == "1")
    review_required = sum(1 for row in ok_records if str(row.get("two_stage_review_required")) == "1")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(args.csv),
        "rows_total": len(records),
        "rows_ok": len(ok_records),
        "rows_failed": len(records) - len(ok_records),
        "device": args.device,
        "settings": {
            "router_min_confidence": args.router_min_confidence,
            "auto_threshold": args.auto_threshold,
            "spam_gate_threshold": args.spam_gate_threshold,
            "spam_gate_allow_threshold": args.spam_gate_allow_threshold,
            "spam_conflict_review_min_confidence": args.spam_conflict_review_min_confidence,
            "nlp_backend": args.nlp_backend,
            "nlp_text_mode": args.nlp_text_mode,
        },
        "models": {
            "single6_model_dir": str(args.single_model_dir),
            "single6_labels": single_model.labels,
            "two_stage_spam_artifact": str(args.spam_artifact_path),
            "two_stage_spam_model_dir": str(args.spam_model_dir),
            "two_stage_stage2_artifact": str(args.stage2_artifact_path),
            "two_stage_stage2_model_dir": str(args.stage2_model_dir),
            "two_stage_stage2_artifact_meta": {
                key: load_torch_artifact(args.stage2_artifact_path).get(key)
                for key in ["artifact_version", "version_id", "trained_at", "model_name", "intent_ids"]
            },
        },
        "two_stage_operational": {
            "auto_ticket_allowed": auto_allowed,
            "review_required": review_required,
            "auto_ticket_allowed_rate": round(auto_allowed / len(ok_records), 6) if ok_records else 0.0,
            "review_required_rate": round(review_required / len(ok_records), 6) if ok_records else 0.0,
        },
        "single6": single_metrics,
        "two_stage": two_stage_metrics,
    }

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else args.csv.parent / "text_routing_predictions.csv"
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else args.csv.parent / "text_routing_metrics.json"
    out_text = Path(args.out_text).expanduser().resolve() if args.out_text else args.csv.parent / "text_routing_metrics.txt"

    write_csv(out_csv, records)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    text_summary = "\n\n".join(
        [
            "Text routing comparison",
            f"rows_ok={len(ok_records)}, rows_failed={len(records) - len(ok_records)}, duration_sec={time.time() - started:.2f}",
            format_report_block("Single 6-class model / all", single_metrics["all"]),
            format_report_block("Two-stage pipeline / all", two_stage_metrics["all"]),
            f"Two-stage operational: auto_ticket_allowed={auto_allowed}, review_required={review_required}",
        ]
    )
    out_text.write_text(text_summary + "\n", encoding="utf-8")

    print("\n" + text_summary)
    print(f"\n[OK] predictions CSV: {out_csv}")
    print(f"[OK] metrics JSON: {out_json}")
    print(f"[OK] metrics TXT: {out_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
