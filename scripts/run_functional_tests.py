#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ALLOWED_AUDIO = {".mp3", ".wav", ".ogg", ".m4a", ".flac"}


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


def make_multipart(file_path):
    boundary = f"----functional-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    parts = [
        f"--{boundary}\r\n".encode("utf-8"),
        f'Content-Disposition: form-data; name="audio"; filename="{file_path.name}"\r\n'.encode("utf-8"),
        f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
        file_bytes,
        b"\r\n",
        f"--{boundary}--\r\n".encode("utf-8"),
    ]
    return b"".join(parts), boundary


def upload_file(url, file_path, headers=None, timeout=3600):
    body, boundary = make_multipart(file_path)
    request_headers = {
        "Accept": "application/json",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
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


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def add_result(results, code, title, requirements, status, note="", details=None):
    results.append(
        {
            "id": code,
            "title": title,
            "requirements": requirements,
            "status": status,
            "error": note,
            "details": details or {},
        }
    )
    if status == "passed":
        print(f"[PASS] {code} {title}")
    elif status == "skipped":
        print(f"[SKIP] {code} {title}: {note}")
    else:
        print(f"[FAIL] {code} {title}: {note}")


def save_markdown(report, path):
    lines = [
        "# Functional Test Report",
        "",
        f"Generated: {report['generated_at']}",
        f"Base URL: {report['base_url']}",
        "",
        f"Passed: {report['summary']['passed']}",
        f"Failed: {report['summary']['failed']}",
        f"Skipped: {report['summary']['skipped']}",
        "",
    ]
    for item in report["results"]:
        reqs = ", ".join(item["requirements"])
        lines.append(f"- {item['id']} | {item['status']} | {item['title']} | {reqs}")
        if item["error"]:
            lines.append(f"  note: {item['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_positive_result(payload, allow_review_status):
    allowed = {"completed"}
    if allow_review_status:
        allowed.update({"awaiting_routing_review", "awaiting_spam_review", "spam_blocked"})

    status = str(payload.get("status") or "").strip()
    check(status in allowed, f"unexpected status: {status}")
    check(str(payload.get("call_id") or "").strip() != "", "call_id is empty")

    transcript = payload.get("transcript")
    check(isinstance(transcript, dict), "transcript is missing")
    segments = transcript.get("segments")
    check(isinstance(segments, list) and len(segments) > 0, "transcript segments are missing")

    routing = payload.get("routing")
    check(isinstance(routing, dict), "routing is missing")
    check(str(routing.get("intent_id") or "").strip() != "", "intent_id is empty")
    check(str(routing.get("priority") or "").strip() != "", "priority is empty")

    times = payload.get("processing_time")
    check(isinstance(times, dict), "processing_time is missing")
    check(times.get("transcription") is not None, "processing_time.transcription is missing")
    check(times.get("routing") is not None, "processing_time.routing is missing")

    entities = payload.get("entities")
    check(isinstance(entities, dict), "entities are missing")

    return {
        "status": status,
        "call_id": str(payload.get("call_id") or ""),
        "segments_count": len(segments),
        "intent_id": str(routing.get("intent_id") or ""),
        "priority": str(routing.get("priority") or ""),
        "group": str(routing.get("suggested_group") or ""),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Simple functional tests for the call-processing subsystem.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--admin-username", required=True)
    parser.add_argument("--admin-password", required=True)
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--operator-username", default="")
    parser.add_argument("--operator-password", default="")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--allow-review-status", action="store_true")
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    audio_files = [Path(x).expanduser().resolve() for x in args.audio]
    if not audio_files:
        print("[ERROR] add at least one --audio file")
        return 2

    for path in audio_files:
        if not path.exists():
            print(f"[ERROR] file not found: {path}")
            return 2
        if path.suffix.lower() not in ALLOWED_AUDIO:
            print(f"[ERROR] unsupported extension for test file: {path}")
            return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else Path.cwd() / "exports" / f"functional_tests_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    call_runs = []
    admin_headers = {}

    try:
        status, body = request_json(
            f"{base_url}/api/v1/auth/login",
            "POST",
            {"username": args.admin_username, "password": args.admin_password},
        )
        check(status == 200, f"admin login returned HTTP {status}")
        check(str(body.get("token") or "").strip() != "", "admin token is missing")
        check(isinstance(body.get("user"), dict), "admin user payload is missing")
        check(str(body["user"].get("role") or "") == "admin", "user role is not admin")
        admin_headers = {"Authorization": f"Bearer {body['token']}"}
        add_result(results, "T1", "Admin login", ["2.3.2.1.2"], "passed", details={"role": "admin"})
    except Exception as exc:
        add_result(results, "T1", "Admin login", ["2.3.2.1.2"], "failed", str(exc))

    try:
        status, body = request_json(f"{base_url}/api/v1/auth/me", headers=admin_headers)
        check(status == 200, f"/auth/me returned HTTP {status}")
        check(str(body.get("role") or "") == "admin", "wrong role in /auth/me")
        add_result(results, "T2", "Check current user", ["2.3.2.1.2"], "passed", details={"role": body.get("role")})
    except Exception as exc:
        add_result(results, "T2", "Check current user", ["2.3.2.1.2"], "failed", str(exc))

    try:
        status, body = request_json(f"{base_url}/api/v1/process-call", "POST")
        check(status == 401, f"expected 401, got {status}")
        check(str(body.get("error") or "") == "unauthorized", "wrong error text")
        add_result(results, "T3", "Request without token", ["2.3.2.1.2", "2.3.2.7.1"], "passed")
    except Exception as exc:
        add_result(results, "T3", "Request without token", ["2.3.2.1.2", "2.3.2.7.1"], "failed", str(exc))

    try:
        status, body = request_json(
            f"{base_url}/api/v1/process-call",
            "POST",
            headers={**admin_headers, "Content-Type": "multipart/form-data; boundary=empty"},
        )
        check(status == 400, f"expected 400, got {status}")
        check(str(body.get("error") or "") == "audio file is required", "wrong error text")
        add_result(results, "T4", "Request without audio", ["2.3.2.1.4"], "passed")
    except Exception as exc:
        add_result(results, "T4", "Request without audio", ["2.3.2.1.4"], "failed", str(exc))

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write("test")
            tmp_path = Path(tmp.name)
        status, body = upload_file(f"{base_url}/api/v1/process-call", tmp_path, headers=admin_headers, timeout=60)
        check(status == 400, f"expected 400, got {status}")
        check("unsupported audio format" in str(body.get("error") or ""), "wrong error text")
        add_result(results, "T5", "Wrong file format", ["2.3.2.1.4"], "passed")
    except Exception as exc:
        add_result(results, "T5", "Wrong file format", ["2.3.2.1.4"], "failed", str(exc))
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)

    for i, audio_path in enumerate(audio_files, start=1):
        try:
            request_id = f"test-{uuid.uuid4().hex}"
            status, body = upload_file(
                f"{base_url}/api/v1/process-call",
                audio_path,
                headers={**admin_headers, "X-Request-ID": request_id},
                timeout=args.timeout,
            )
            check(status == 200, f"expected 200, got {status}")
            info = validate_positive_result(body, args.allow_review_status)
            info["request_id"] = request_id
            info["audio_path"] = str(audio_path)
            info["payload"] = body
            call_runs.append(info)
            add_result(
                results,
                f"T{5 + i}",
                f"Process {audio_path.suffix.lower()} file",
                ["2.3.2.1.1", "2.3.2.2.1", "2.3.2.3.1"],
                "passed",
                details={k: v for k, v in info.items() if k != "payload"},
            )
        except Exception as exc:
            add_result(
                results,
                f"T{5 + i}",
                f"Process {audio_path.suffix.lower()} file",
                ["2.3.2.1.1", "2.3.2.2.1", "2.3.2.3.1"],
                "failed",
                str(exc),
            )

    completed_call = None
    for item in call_runs:
        if item["status"] == "completed":
            completed_call = item
            break

    if completed_call is None:
        add_result(
            results,
            "T20",
            "Ticket creation after full processing",
            ["2.3.2.4.4", "2.3.2.4.5"],
            "skipped",
            "no completed call in test set",
        )
        add_result(
            results,
            "T21",
            "Notification after full processing",
            ["2.3.2.5.1", "2.3.2.5.3"],
            "skipped",
            "no completed call in test set",
        )
    else:
        try:
            ticket = completed_call["payload"].get("ticket")
            check(isinstance(ticket, dict), "ticket is missing")
            check(str(ticket.get("ticket_id") or "").strip() != "", "ticket_id is empty")
            check(str(ticket.get("url") or "").strip() != "", "ticket url is empty")
            add_result(
                results,
                "T20",
                "Ticket creation after full processing",
                ["2.3.2.4.4", "2.3.2.4.5"],
                "passed",
                details={"ticket_id": ticket.get("ticket_id")},
            )
        except Exception as exc:
            add_result(results, "T20", "Ticket creation after full processing", ["2.3.2.4.4", "2.3.2.4.5"], "failed", str(exc))

        try:
            notification = completed_call["payload"].get("notification")
            check(isinstance(notification, dict), "notification is missing")
            check(isinstance(notification.get("channels"), list), "notification channels are missing")
            add_result(
                results,
                "T21",
                "Notification after full processing",
                ["2.3.2.5.1", "2.3.2.5.3"],
                "passed",
                details={"channels": len(notification.get("channels") or [])},
            )
        except Exception as exc:
            add_result(results, "T21", "Notification after full processing", ["2.3.2.5.1", "2.3.2.5.3"], "failed", str(exc))

    try:
        check(len(call_runs) > 0, "no successful requests to check in audit")
        target_request_id = call_runs[0]["request_id"]
        status, body = request_json(
            f"{base_url}/api/v1/audit/events?limit=200&event_type=call.process&outcome=success",
            headers=admin_headers,
        )
        check(status == 200, f"audit endpoint returned HTTP {status}")
        events = body.get("events")
        check(isinstance(events, list), "audit events list is missing")
        found = False
        for event in events:
            if isinstance(event, dict) and str(event.get("request_id") or "") == target_request_id:
                found = True
                break
        check(found, "request was not found in audit")
        add_result(results, "T22", "Audit log", ["2.3.2.6.1", "2.3.2.7.3"], "passed")
    except Exception as exc:
        add_result(results, "T22", "Audit log", ["2.3.2.6.1", "2.3.2.7.3"], "failed", str(exc))

    if args.operator_username and args.operator_password:
        try:
            status, body = request_json(
                f"{base_url}/api/v1/auth/login",
                "POST",
                {"username": args.operator_username, "password": args.operator_password},
            )
            check(status == 200, f"operator login returned HTTP {status}")
            check(str(body.get("token") or "").strip() != "", "operator token is missing")
            headers = {"Authorization": f"Bearer {body['token']}"}
            status, body = request_json(f"{base_url}/api/v1/audit/events", headers=headers)
            check(status == 403, f"expected 403, got {status}")
            check(str(body.get("error") or "") == "forbidden", "wrong error text")
            add_result(results, "T23", "Operator has no admin access", ["2.3.2.1.3", "2.3.2.7.1"], "passed")
        except Exception as exc:
            add_result(results, "T23", "Operator has no admin access", ["2.3.2.1.3", "2.3.2.7.1"], "failed", str(exc))
    else:
        add_result(
            results,
            "T23",
            "Operator has no admin access",
            ["2.3.2.1.3", "2.3.2.7.1"],
            "skipped",
            "operator credentials were not provided",
        )

    passed = sum(1 for x in results if x["status"] == "passed")
    failed = sum(1 for x in results if x["status"] == "failed")
    skipped = sum(1 for x in results if x["status"] == "skipped")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "audio_files": [str(x) for x in audio_files],
        "summary": {"total": len(results), "passed": passed, "failed": failed, "skipped": skipped},
        "results": results,
    }

    json_path = output_dir / "functional_test_report.json"
    md_path = output_dir / "functional_test_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_markdown(report, md_path)

    print("\n[DONE]")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"[OK] JSON report: {json_path}")
    print(f"[OK] Markdown report: {md_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
