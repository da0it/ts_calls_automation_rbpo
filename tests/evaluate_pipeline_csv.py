#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from pathlib import Path


ALLOWED_AUDIO = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


def detect_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="ignore")[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return dialect.delimiter
    except Exception:
        if sample.count(";") >= sample.count(","):
            return ";"
        return ","


def read_rows(path: Path):
    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=delimiter)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers, delimiter


def request_json(url, method="GET", payload=None, headers=None, timeout=60):
    body = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers["Content-Type"] = "application/json; charset=utf-8"

    request = urllib.request.Request(url=url, data=body, method=method.upper(), headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc

    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {"raw": raw}
    return status, data


def upload_file(url, file_path: Path, headers=None, timeout=3600):
    boundary = f"----pipeline-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="audio"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
    )

    request_headers = {"Accept": "application/json", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        request_headers.update(headers)

    request = urllib.request.Request(url=url, data=body, method="POST", headers=request_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        status = exc.code
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc

    try:
        data = json.loads(raw) if raw else {}
    except Exception:
        data = {"raw": raw}
    return status, data


def clean(value) -> str:
    return str(value or "").strip()


def find_audio_path(row, csv_dir: Path, audio_dir: Path | None, indexed_files: dict[str, Path]):
    for key in ["audio_path", "file_path", "path"]:
        raw = clean(row.get(key))
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
        if not path.is_absolute():
            local = (csv_dir / raw).resolve()
            if local.exists():
                return local

    filename = clean(row.get("filename"))
    if not filename:
        return None

    candidate = Path(filename).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        local = (csv_dir / filename).resolve()
        if local.exists():
            return local

    if audio_dir:
        direct = (audio_dir / filename).resolve()
        if direct.exists():
            return direct
        by_name = indexed_files.get(Path(filename).name)
        if by_name is not None:
            return by_name

    return None


def to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def calc_binary_metrics(rows):
    tp = fp = tn = fn = 0
    for row in rows:
        gold = row["gold_binary"]
        pred = row["pred_binary"]
        if gold == "spam" and pred == "spam":
            tp += 1
        elif gold == "non_spam" and pred == "spam":
            fp += 1
        elif gold == "non_spam" and pred == "non_spam":
            tn += 1
        elif gold == "spam" and pred == "non_spam":
            fn += 1

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "samples": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": round(accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def build_summary(results):
    processed = [x for x in results if not x["error"]]
    binary_rows = [x for x in processed if x["gold_binary"] in {"spam", "non_spam"}]
    nonspam_rows = [x for x in processed if x["gold_binary"] == "non_spam"]

    exact_intent_matches = 0
    for row in nonspam_rows:
        if row["predicted_intent"] == row["gold_call_purpose"]:
            exact_intent_matches += 1

    total_time_values = [x["total_time"] for x in processed if x["total_time"] is not None]
    confidence_values = [x["predicted_confidence"] for x in processed if x["predicted_confidence"] is not None]

    return {
        "rows_total": len(results),
        "processed_ok": len(processed),
        "processed_with_error": sum(1 for x in results if x["error"]),
        "missing_audio": sum(1 for x in results if x["error"] == "audio_not_found"),
        "status_counts": dict(Counter(x["pipeline_status"] or "unknown" for x in processed)),
        "spam_binary": calc_binary_metrics(binary_rows),
        "nonspam_intent": {
            "samples": len(nonspam_rows),
            "exact_matches": exact_intent_matches,
            "accuracy": round(exact_intent_matches / len(nonspam_rows), 6) if nonspam_rows else 0.0,
        },
        "avg_total_time_sec": round(sum(total_time_values) / len(total_time_values), 6) if total_time_values else None,
        "avg_confidence": round(sum(confidence_values) / len(confidence_values), 6) if confidence_values else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run CSV rows through /api/v1/process-call and save results.")
    parser.add_argument("--csv", default="final.csv")
    parser.add_argument("--audio-dir", default="")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--out-csv", default="")
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else None
    if audio_dir and not audio_dir.exists():
        print(f"[ERROR] audio dir not found: {audio_dir}")
        return 2

    rows, headers, _ = read_rows(csv_path)
    if not rows:
        print(f"[ERROR] csv is empty: {csv_path}")
        return 2
    if "call_purpose" not in headers:
        print("[ERROR] column call_purpose is required")
        return 2

    index = {}
    if audio_dir:
        for path in audio_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_AUDIO:
                index[path.name] = path.resolve()

    base_url = args.base_url.rstrip("/")
    status, body = request_json(
        f"{base_url}/api/v1/auth/login",
        "POST",
        {"username": args.username, "password": args.password},
    )
    if status != 200 or not clean(body.get("token")):
        print(f"[ERROR] login failed: status={status}")
        return 2
    headers_auth = {"Authorization": f"Bearer {body['token']}"}

    results = []
    csv_dir = csv_path.parent

    for idx, row in enumerate(rows, start=1):
        gold = clean(row.get("call_purpose"))
        gold_binary = "spam" if gold.lower() == "spam" else "non_spam"
        audio_path = find_audio_path(row, csv_dir, audio_dir, index)

        item = {
            "row_id": idx,
            "filename": clean(row.get("filename")),
            "audio_path": str(audio_path) if audio_path else "",
            "gold_call_purpose": gold,
            "gold_binary": gold_binary,
            "http_status": None,
            "pipeline_status": "",
            "pred_binary": "",
            "predicted_intent": "",
            "predicted_group": "",
            "predicted_priority": "",
            "predicted_confidence": None,
            "call_id": "",
            "ticket_id": "",
            "ticket_url": "",
            "total_time": None,
            "binary_match": "",
            "intent_match_nonspam": "",
            "error": "",
        }

        if audio_path is None:
            item["error"] = "audio_not_found"
            results.append(item)
            print(f"[SKIP] row {idx}: audio not found")
            continue

        try:
            request_id = f"dataset-{uuid.uuid4().hex}"
            status, payload = upload_file(
                f"{base_url}/api/v1/process-call",
                audio_path,
                headers={**headers_auth, "X-Request-ID": request_id},
                timeout=args.timeout,
            )
            item["http_status"] = status
            if status != 200:
                item["error"] = f"http_{status}"
                results.append(item)
                print(f"[FAIL] row {idx}: http {status}")
                continue

            routing = payload.get("routing") or {}
            ticket = payload.get("ticket") or {}
            item["pipeline_status"] = clean(payload.get("status"))
            item["predicted_intent"] = clean(routing.get("intent_id"))
            item["predicted_group"] = clean(routing.get("suggested_group"))
            item["predicted_priority"] = clean(routing.get("priority"))
            item["predicted_confidence"] = to_float(routing.get("intent_confidence"))
            item["call_id"] = clean(payload.get("call_id"))
            item["ticket_id"] = clean(ticket.get("external_id") or ticket.get("ticket_id"))
            item["ticket_url"] = clean(ticket.get("url"))
            item["total_time"] = to_float(payload.get("total_time"))

            pred_binary = "spam" if item["pipeline_status"] == "spam_blocked" or item["predicted_intent"] in {"spam", "spam.call"} else "non_spam"
            item["pred_binary"] = pred_binary
            item["binary_match"] = "1" if pred_binary == gold_binary else "0"
            if gold_binary == "non_spam":
                item["intent_match_nonspam"] = "1" if item["predicted_intent"] == gold else "0"

            results.append(item)
            print(
                f"[OK] row {idx}: status={item['pipeline_status']} "
                f"gold={gold} pred={item['predicted_intent'] or '-'}"
            )
        except Exception as exc:
            item["error"] = str(exc)
            results.append(item)
            print(f"[FAIL] row {idx}: {exc}")

    summary = build_summary(results)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "csv": str(csv_path),
        "audio_dir": str(audio_dir) if audio_dir else "",
        "base_url": base_url,
        "summary": summary,
    }

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else (csv_path.parent / "pipeline_results.csv")
    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (csv_path.parent / "pipeline_summary.json")

    fieldnames = list(rows[0].keys()) + [
        "row_id",
        "audio_path",
        "gold_call_purpose",
        "gold_binary",
        "http_status",
        "pipeline_status",
        "pred_binary",
        "predicted_intent",
        "predicted_group",
        "predicted_priority",
        "predicted_confidence",
        "call_id",
        "ticket_id",
        "ticket_url",
        "total_time",
        "binary_match",
        "intent_match_nonspam",
        "error",
    ]
    fieldnames = list(dict.fromkeys(fieldnames))

    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for source, extra in zip(rows, results):
            merged = dict(source)
            merged.update(extra)
            writer.writerow(merged)

    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
