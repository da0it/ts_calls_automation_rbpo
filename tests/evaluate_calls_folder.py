#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ALLOWED_DEFAULT_EXTS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]


def _http_json(
    url: str,
    method: str,
    payload: Dict[str, object] | None = None,
    headers: Dict[str, str] | None = None,
    timeout: int = 60,
) -> Dict[str, object]:
    body = None
    req_headers = {"Accept": "application/json"}
    if headers:
        req_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req_headers["Content-Type"] = "application/json; charset=utf-8"
    req = urllib.request.Request(url=url, data=body, method=method.upper(), headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: HTTP {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _encode_multipart_form(file_field: str, file_path: Path) -> Tuple[bytes, str]:
    boundary = f"----eval-{uuid.uuid4().hex}"
    chunks: List[bytes] = []

    file_name = file_path.name
    content_type = mimetypes.guess_type(file_name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    chunks.append(f"--{boundary}\r\n".encode("utf-8"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{file_name}"\r\n'
        ).encode("utf-8")
    )
    chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    chunks.append(file_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), boundary


def _http_multipart_json(
    url: str,
    file_field: str,
    file_path: Path,
    headers: Dict[str, str] | None = None,
    timeout: int = 3600,
) -> Dict[str, object]:
    body, boundary = _encode_multipart_form(file_field=file_field, file_path=file_path)
    req_headers = {"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url=url, data=body, method="POST", headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed for {file_path.name}: HTTP {exc.code}: {raw}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {url} failed for {file_path.name}: {exc}") from exc


def _collect_audio_files(root: Path, exts: Iterable[str], recursive: bool) -> List[Path]:
    allowed = {ext.lower() for ext in exts}
    if recursive:
        files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in allowed]
    else:
        files = [path for path in root.glob("*") if path.is_file() and path.suffix.lower() in allowed]
    return sorted(files)


def _join_segments_text(segments: List[Dict[str, object]]) -> str:
    parts: List[str] = []
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _infer_true_intent(file_path: Path, input_dir: Path, label_mode: str) -> str:
    if label_mode == "none":
        return ""
    if label_mode == "parent_dir":
        try:
            rel = file_path.relative_to(input_dir)
        except ValueError:
            return ""
        if len(rel.parts) >= 2:
            return str(rel.parts[0]).strip()
    return ""


def _load_labels_csv(path: Path, path_col: str, label_cols: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = list(reader.fieldnames or [])
        if path_col not in headers:
            raise RuntimeError(f"labels csv must contain path column {path_col!r}")
        resolved_label_cols = {
            target_key: column_name
            for target_key, column_name in label_cols.items()
            if column_name and column_name in headers
        }
        if not resolved_label_cols:
            requested = [repr(name) for name in label_cols.values() if name]
            raise RuntimeError(
                "labels csv does not contain any of the requested label columns: "
                + ", ".join(requested)
            )
        for row in reader:
            key = str(row.get(path_col) or "").strip()
            if not key:
                continue
            item: Dict[str, str] = {}
            for target_key, column_name in resolved_label_cols.items():
                item[target_key] = str(row.get(column_name) or "").strip()
            out[key] = item
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate audio calls from a folder via orchestrator /api/v1/process-call."
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Orchestrator base URL.")
    parser.add_argument("--username", default="", help="Username for /api/v1/auth/login.")
    parser.add_argument("--password", default="", help="Password for /api/v1/auth/login.")
    parser.add_argument("--token", default="", help="Bearer token. If set, login is skipped.")
    parser.add_argument("--input-dir", required=True, help="Directory with audio files.")
    parser.add_argument("--output-dir", default="", help="Output directory (default: ./exports/call_eval_<timestamp>).")
    parser.add_argument("--extensions", default=",".join(ALLOWED_DEFAULT_EXTS), help="Comma-separated audio extensions.")
    parser.add_argument("--no-recursive", action="store_true", help="Do not scan subdirectories.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0=all).")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-file request timeout in seconds.")
    parser.add_argument("--label-mode", choices=["none", "parent_dir"], default="none")
    parser.add_argument("--labels-csv", default="", help="Optional CSV with true labels.")
    parser.add_argument("--labels-path-col", default="path", help="Column in labels CSV with relative file path.")
    parser.add_argument("--labels-intent-col", default="intent_id", help="Column in labels CSV with true intent.")
    parser.add_argument("--labels-spam-col", default="", help="Optional column in labels CSV with manual spam flag.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first failed file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] input dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path.cwd() / "exports" / f"call_eval_{time.strftime('%Y%m%d_%H%M%S')}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = [item.strip().lower() for item in args.extensions.split(",") if item.strip()]
    files = _collect_audio_files(input_dir, exts=exts, recursive=not args.no_recursive)
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        print(f"[WARN] no audio files found in {input_dir} with extensions {exts}")
        return 0

    labels_map: Dict[str, Dict[str, str]] = {}
    if args.labels_csv:
        labels_map = _load_labels_csv(
            Path(args.labels_csv).expanduser().resolve(),
            path_col=str(args.labels_path_col),
            label_cols={
                "true_intent": str(args.labels_intent_col or "").strip(),
                "manual_is_spam": str(args.labels_spam_col or "").strip(),
            },
        )

    token = str(args.token or "").strip()
    if not token:
        if not args.username or not args.password:
            print("[ERROR] provide either --token or both --username and --password", file=sys.stderr)
            return 2
        print("[INFO] authenticating...")
        login = _http_json(
            f"{base_url}/api/v1/auth/login",
            "POST",
            payload={"username": args.username, "password": args.password},
            timeout=60,
        )
        token = str(login.get("token") or "").strip()
        if not token:
            print("[ERROR] login succeeded but token is missing", file=sys.stderr)
            return 2
    headers = {"Authorization": f"Bearer {token}"}

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []
    pred_counter: Counter[str] = Counter()
    true_counter: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = defaultdict(Counter)

    for idx, path in enumerate(files, start=1):
        rel = path.relative_to(input_dir)
        rel_str = rel.as_posix()
        label_item = labels_map.get(rel_str, {})
        true_intent = str(label_item.get("true_intent") or "") or _infer_true_intent(path, input_dir, str(args.label_mode))
        manual_is_spam = str(label_item.get("manual_is_spam") or "").strip()
        print(f"[{idx}/{len(files)}] {rel_str}")
        try:
            payload = _http_multipart_json(
                f"{base_url}/api/v1/process-call",
                file_field="audio",
                file_path=path,
                headers=headers,
                timeout=int(args.timeout),
            )
            transcript = payload.get("transcript") if isinstance(payload.get("transcript"), dict) else {}
            routing = payload.get("routing") if isinstance(payload.get("routing"), dict) else {}
            spam_check = payload.get("spam_check") if isinstance(payload.get("spam_check"), dict) else {}
            processing_time = payload.get("processing_time") if isinstance(payload.get("processing_time"), dict) else {}

            segments = [seg for seg in (transcript.get("segments") or []) if isinstance(seg, dict)]
            pred_intent = str(routing.get("intent_id") or "").strip()
            confidence = _safe_float(routing.get("intent_confidence"))
            row = {
                "source_file": rel_str,
                "filename": path.name,
                "true_intent": true_intent,
                "manual_is_spam": manual_is_spam,
                "status": str(payload.get("status") or ""),
                "pred_intent": pred_intent,
                "confidence": round(confidence, 6),
                "priority": str(routing.get("priority") or ""),
                "suggested_group": str(routing.get("suggested_group") or ""),
                "spam_status": str(spam_check.get("status") or ""),
                "spam_predicted_label": str(spam_check.get("predicted_label") or ""),
                "spam_confidence": round(_safe_float(spam_check.get("confidence")), 6),
                "spam_reason": str(spam_check.get("reason") or ""),
                "is_triage": int(pred_intent == "misc.triage"),
                "is_zero_confidence": int(confidence == 0.0),
                "is_correct": "",
                "call_id": str(payload.get("call_id") or transcript.get("call_id") or ""),
                "segments_count": len(segments),
                "transcript_text": _join_segments_text(segments),
                "processing_total_sec": round(_safe_float(payload.get("total_time")), 4),
                "processing_transcription_sec": round(_safe_float(processing_time.get("transcription")), 4),
                "processing_routing_sec": round(_safe_float(processing_time.get("routing")), 4),
            }
            if true_intent:
                is_correct = int(pred_intent == true_intent)
                row["is_correct"] = is_correct
                true_counter[true_intent] += 1
                confusion[true_intent][pred_intent] += 1
            pred_counter[pred_intent] += 1
            rows.append(row)
        except Exception as exc:
            msg = str(exc)
            print(f"  [ERROR] {msg}", file=sys.stderr)
            failures.append({"source_file": rel_str, "error": msg})
            if args.stop_on_error:
                break

    results_csv = out_dir / "results.csv"
    failures_csv = out_dir / "failures.csv"
    summary_json = out_dir / "summary.json"
    raw_jsonl = out_dir / "results.jsonl"

    fieldnames = [
        "source_file",
        "filename",
        "true_intent",
        "manual_is_spam",
        "status",
        "pred_intent",
        "confidence",
        "priority",
        "suggested_group",
        "spam_status",
        "spam_predicted_label",
        "spam_confidence",
        "spam_reason",
        "is_triage",
        "is_zero_confidence",
        "is_correct",
        "call_id",
        "segments_count",
        "processing_total_sec",
        "processing_transcription_sec",
        "processing_routing_sec",
        "transcript_text",
    ]
    with results_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        with failures_csv.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["source_file", "error"])
            writer.writeheader()
            writer.writerows(failures)

    with raw_jsonl.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    confidences = [float(row["confidence"]) for row in rows]
    triage_count = sum(int(row["is_triage"]) for row in rows)
    zero_count = sum(int(row["is_zero_confidence"]) for row in rows)
    labeled_rows = [row for row in rows if str(row.get("true_intent") or "").strip()]
    correct_count = sum(int(row.get("is_correct") or 0) for row in labeled_rows)
    manual_spam_rows = [row for row in rows if str(row.get("manual_is_spam") or "").strip() != ""]

    status_counts: Counter[str] = Counter(str(row.get("status") or "").strip() or "unknown" for row in rows)
    spam_status_counts: Counter[str] = Counter(str(row.get("spam_status") or "").strip() or "unknown" for row in rows)
    spam_confusion: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in manual_spam_rows:
        true_label = "spam" if str(row.get("manual_is_spam") or "").strip() in {"1", "true", "yes", "spam"} else "not_spam"
        pred_status = str(row.get("spam_status") or "").strip() or "unknown"
        spam_confusion[true_label][pred_status] += 1

    confidence_bands = {
        "lt_0_35": sum(1 for value in confidences if value < 0.35),
        "0_35_to_0_55": sum(1 for value in confidences if 0.35 <= value < 0.55),
        "ge_0_55": sum(1 for value in confidences if value >= 0.55),
    }

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "total_files": len(files),
        "processed_ok": len(rows),
        "failed": len(failures),
        "labeled_files": len(labeled_rows),
        "spam_labeled_files": len(manual_spam_rows),
        "accuracy": round(correct_count / len(labeled_rows), 4) if labeled_rows else None,
        "triage_rate": round(triage_count / len(rows), 4) if rows else None,
        "zero_confidence_rate": round(zero_count / len(rows), 4) if rows else None,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "confidence_bands": confidence_bands,
        "status_counts": dict(sorted(status_counts.items())),
        "spam_status_counts": dict(sorted(spam_status_counts.items())),
        "spam_confusion": {key: dict(sorted(value.items())) for key, value in sorted(spam_confusion.items())},
        "predicted_intents": dict(sorted(pred_counter.items())),
        "true_intents": dict(sorted(true_counter.items())),
        "confusion": {key: dict(sorted(value.items())) for key, value in sorted(confusion.items())},
        "results_csv": str(results_csv),
        "results_jsonl": str(raw_jsonl),
        "failures_csv": str(failures_csv) if failures else "",
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
