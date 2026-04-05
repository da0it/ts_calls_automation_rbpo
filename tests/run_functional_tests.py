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
REQUIRED_AUDIO_FOR_FULL_COVERAGE = {".mp3", ".wav", ".ogg"}
KNOWN_PROCESS_STATUSES = {"completed", "awaiting_routing_review", "spam_blocked"}
ENTITY_KEYS = [
    "persons",
    "phones",
    "emails",
    "order_ids",
    "account_ids",
    "money_amounts",
    "dates",
]


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


def upload_file(url, file_path, headers=None, timeout=3600):
    boundary = f"----functional-{uuid.uuid4().hex}"
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


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def add_result(results, code, title, status, note="", details=None):
    results.append(
        {
            "id": code,
            "title": title,
            "status": status,
            "error": note,
            "details": details or {},
        }
    )
    prefix = {"passed": "[PASS]", "failed": "[FAIL]", "skipped": "[SKIP]"}[status]
    text = f"{prefix} {code} {title}"
    if note:
        text += f": {note}"
    print(text)


def get_result_status(results, code):
    for item in results:
        if item["id"] == code:
            return item["status"]
    return ""


def mark_requirement(results, code, title, ok, note="", details=None):
    add_result(results, code, title, "passed" if ok else "failed", note, details)


def validate_ok_response(payload, allow_review_status):
    status = str(payload.get("status") or "").strip()
    require(status in KNOWN_PROCESS_STATUSES, f"unexpected status: {status}")
    require(str(payload.get("call_id") or "").strip() != "", "call_id is empty")
    require(isinstance(payload.get("transcript"), dict), "transcript is missing")
    require(isinstance(payload["transcript"].get("segments"), list), "segments are missing")
    require(len(payload["transcript"]["segments"]) > 0, "segments are empty")
    for segment in payload["transcript"]["segments"][:3]:
        require(isinstance(segment, dict), "segment must be an object")
        require(str(segment.get("text") or "").strip() != "", "segment text is empty")
        require(segment.get("start") is not None, "segment start is missing")
        require(segment.get("end") is not None, "segment end is missing")
    require(isinstance(payload.get("routing"), dict), "routing is missing")
    require(str(payload["routing"].get("intent_id") or "").strip() != "", "intent_id is empty")
    require(payload["routing"].get("intent_confidence") is not None, "intent_confidence is empty")
    require(str(payload["routing"].get("priority") or "").strip() != "", "priority is empty")
    require(str(payload["routing"].get("suggested_group") or "").strip() != "", "suggested_group is empty")
    require(isinstance(payload.get("processing_time"), dict), "processing_time is missing")
    require(payload["processing_time"].get("transcription") is not None, "transcription time is missing")
    require(payload["processing_time"].get("routing") is not None, "routing time is missing")
    require(isinstance(payload.get("entities"), dict), "entities are missing")
    for key in ENTITY_KEYS:
        require(isinstance(payload["entities"].get(key), list), f"entities.{key} is missing")

    return {
        "status": status,
        "call_id": str(payload.get("call_id") or ""),
        "queue_id": str(payload.get("queue_id") or ""),
        "intent_id": str(payload["routing"].get("intent_id") or ""),
        "intent_confidence": float(payload["routing"].get("intent_confidence") or 0.0),
        "priority": str(payload["routing"].get("priority") or ""),
        "suggested_group": str(payload["routing"].get("suggested_group") or ""),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Simple functional tests for the HTTP API.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--admin-username", required=True)
    parser.add_argument("--admin-password", required=True)
    parser.add_argument("--audio", action="append", default=[])
    parser.add_argument("--operator-username", default="")
    parser.add_argument("--operator-password", default="")
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--allow-review-status", action="store_true")
    parser.add_argument("--out-json", default="")
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
            print(f"[ERROR] unsupported test file: {path}")
            return 2

    results = []
    good_calls = []
    admin_headers = {}
    operator_headers = {}
    queue_snapshot = None
    audit_snapshot = None

    def run(code, title, fn):
        try:
            details = fn() or {}
            add_result(results, code, title, "passed", details=details)
        except Exception as exc:
            add_result(results, code, title, "failed", str(exc))

    def test_admin_login():
        nonlocal admin_headers
        status, body = request_json(
            f"{base_url}/api/v1/auth/login",
            "POST",
            {"username": args.admin_username, "password": args.admin_password},
        )
        require(status == 200, f"expected 200, got {status}")
        require(str(body.get("token") or "").strip() != "", "token is missing")
        require(isinstance(body.get("user"), dict), "user is missing")
        require(str(body["user"].get("role") or "") == "admin", "role is not admin")
        admin_headers = {"Authorization": f"Bearer {body['token']}"}
        return {"role": "admin"}

    def test_me():
        status, body = request_json(f"{base_url}/api/v1/auth/me", headers=admin_headers)
        require(status == 200, f"expected 200, got {status}")
        require(str(body.get("role") or "") == "admin", "wrong role")

    def test_no_token():
        status, body = request_json(f"{base_url}/api/v1/process-call", "POST")
        require(status == 401, f"expected 401, got {status}")
        require(str(body.get("error") or "") == "unauthorized", "wrong error text")

    def test_no_audio():
        status, body = request_json(
            f"{base_url}/api/v1/process-call",
            "POST",
            headers={**admin_headers, "Content-Type": "multipart/form-data; boundary=empty"},
        )
        require(status == 400, f"expected 400, got {status}")
        require(str(body.get("error") or "") == "audio file is required", "wrong error text")

    def test_bad_format():
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write("test")
            tmp_path = Path(tmp.name)
        try:
            status, body = upload_file(f"{base_url}/api/v1/process-call", tmp_path, headers=admin_headers, timeout=60)
            require(status == 400, f"expected 400, got {status}")
            require("unsupported audio format" in str(body.get("error") or ""), "wrong error text")
        finally:
            tmp_path.unlink(missing_ok=True)

    def ensure_operator():
        nonlocal operator_headers
        username = args.operator_username.strip()
        password = args.operator_password.strip()

        if not username or not password:
            username = f"operator_{uuid.uuid4().hex[:10]}"
            password = f"pass_{uuid.uuid4().hex[:10]}"
            status, body = request_json(
                f"{base_url}/api/v1/users",
                "POST",
                {"username": username, "password": password, "role": "operator"},
                headers=admin_headers,
            )
            require(status == 201, f"expected 201, got {status}")
            require(str(body.get("role") or "") == "operator", "created user role is not operator")

        status, body = request_json(
            f"{base_url}/api/v1/auth/login",
            "POST",
            {"username": username, "password": password},
        )
        require(status == 200, f"expected 200, got {status}")
        require(str(body.get("token") or "").strip() != "", "operator token is missing")
        require(isinstance(body.get("user"), dict), "operator user is missing")
        require(str(body["user"].get("role") or "") == "operator", "operator role is wrong")
        operator_headers = {"Authorization": f"Bearer {body['token']}"}
        return {"username": username}

    def test_operator_process_access():
        ensure_operator()
        status, body = request_json(
            f"{base_url}/api/v1/process-call",
            "POST",
            headers={**operator_headers, "Content-Type": "multipart/form-data; boundary=empty"},
        )
        require(status == 400, f"expected 400, got {status}")
        require(str(body.get("error") or "") == "audio file is required", "wrong error text")

    def test_operator_forbidden():
        ensure_operator()
        status, body = request_json(f"{base_url}/api/v1/audit/events", headers=operator_headers)
        require(status == 403, f"expected 403, got {status}")
        require(str(body.get("error") or "") == "forbidden", "wrong error text")

    run("T1", "Admin login", test_admin_login)
    run("T2", "Check current user", test_me)
    run("T3", "Request without token", test_no_token)
    run("T4", "Request without audio", test_no_audio)
    run("T5", "Wrong file format", test_bad_format)

    for index, audio_path in enumerate(audio_files, start=1):
        try:
            request_id = f"test-{uuid.uuid4().hex}"
            status, body = upload_file(
                f"{base_url}/api/v1/process-call",
                audio_path,
                headers={**admin_headers, "X-Request-ID": request_id},
                timeout=args.timeout,
            )
            require(status == 200, f"expected 200, got {status}")
            info = validate_ok_response(body, args.allow_review_status)
            info["request_id"] = request_id
            info["payload"] = body
            info["audio_ext"] = audio_path.suffix.lower()
            good_calls.append(info)
            add_result(results, f"T{5 + index}", f"Process {audio_path.suffix.lower()} file", "passed", details=info)
        except Exception as exc:
            add_result(results, f"T{5 + index}", f"Process {audio_path.suffix.lower()} file", "failed", str(exc))

    completed_call = next((x for x in good_calls if x["status"] == "completed"), None)

    if completed_call is None:
        add_result(results, "T20", "Ticket creation", "skipped", "no completed call")
    else:
        try:
            ticket = completed_call["payload"].get("ticket")
            require(isinstance(ticket, dict), "ticket is missing")
            require(str(ticket.get("ticket_id") or "").strip() != "", "ticket_id is empty")
            require(str(ticket.get("url") or "").strip() != "", "ticket url is empty")
            add_result(results, "T20", "Ticket creation", "passed", details={"ticket_id": ticket["ticket_id"]})
        except Exception as exc:
            add_result(results, "T20", "Ticket creation", "failed", str(exc))

    try:
        require(good_calls, "no successful process-call requests")
        queue_call = good_calls[0]
        require(queue_call["queue_id"], "queue_id is empty")
        status, body = request_json(f"{base_url}/api/v1/calls?limit=200", headers=admin_headers)
        require(status == 200, f"expected 200, got {status}")
        require(isinstance(body.get("calls"), list), "calls are missing")

        found = None
        for item in body["calls"]:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            item_call_id = str(item.get("callId") or "").strip()
            if item_id == queue_call["queue_id"] or item_call_id == queue_call["call_id"]:
                found = item
                break

        require(found is not None, "processed call was not found in shared queue")
        require(str(found.get("callId") or "").strip() == queue_call["call_id"], "wrong call id in shared queue")
        require(str(found.get("status") or "").strip() == queue_call["status"], "wrong status in shared queue")
        require(isinstance(found.get("raw"), dict), "raw processing report is missing in shared queue")
        require(isinstance(found["raw"].get("routing"), dict), "raw routing is missing in shared queue")
        require(isinstance(found["raw"].get("transcript"), dict), "raw transcript is missing in shared queue")
        require(isinstance(found.get("spamCheck"), dict), "spamCheck is missing in shared queue")
        queue_snapshot = found
        add_result(results, "T21", "Shared call queue", "passed", details={"queue_id": queue_call["queue_id"]})
    except Exception as exc:
        add_result(results, "T21", "Shared call queue", "failed", str(exc))

    try:
        require(good_calls, "no successful process-call requests")
        status, body = request_json(
            f"{base_url}/api/v1/audit/events?limit=200&event_type=call.process&outcome=success",
            headers=admin_headers,
        )
        require(status == 200, f"expected 200, got {status}")
        require(isinstance(body.get("events"), list), "events are missing")
        found = None
        for item in body["events"]:
            if not isinstance(item, dict):
                continue
            if str(item.get("request_id") or "") == good_calls[0]["request_id"]:
                found = item
                break
        require(found is not None, "request was not found in audit")
        require(isinstance(found.get("details"), dict), "audit details are missing")
        audit_snapshot = found
        add_result(results, "T22", "Audit log", "passed")
    except Exception as exc:
        add_result(results, "T22", "Audit log", "failed", str(exc))

    run("T23", "Operator can access process endpoint", test_operator_process_access)
    run("T24", "Operator has no admin access", test_operator_forbidden)

    if completed_call is None:
        add_result(results, "T25", "Notification handoff to external ticket system", "skipped", "no completed call")
    else:
        try:
            routing = completed_call["payload"].get("routing") or {}
            ticket = completed_call["payload"].get("ticket") or {}
            require(str(routing.get("suggested_group") or "").strip() != "", "suggested_group is empty")
            require(str(ticket.get("system") or "").strip() != "", "ticket system is empty")
            require(
                str(ticket.get("external_id") or ticket.get("ticket_id") or "").strip() != "",
                "ticket external id is empty",
            )
            add_result(
                results,
                "T25",
                "Notification handoff to external ticket system",
                "passed",
                details={"group": routing.get("suggested_group"), "system": ticket.get("system")},
            )
        except Exception as exc:
            add_result(results, "T25", "Notification handoff to external ticket system", "failed", str(exc))

    processed_exts = {item["audio_ext"] for item in good_calls}
    missing_required_exts = sorted(REQUIRED_AUDIO_FOR_FULL_COVERAGE - processed_exts)
    mark_requirement(
        results,
        "T30",
        "Requirement 1: Intake, validation and persistence",
        get_result_status(results, "T1") == "passed"
        and get_result_status(results, "T3") == "passed"
        and get_result_status(results, "T4") == "passed"
        and get_result_status(results, "T5") == "passed"
        and get_result_status(results, "T21") == "passed"
        and not missing_required_exts,
        "" if not missing_required_exts else f"full format coverage still needs: {', '.join(missing_required_exts)}",
        {"processed_formats": sorted(processed_exts)},
    )
    mark_requirement(
        results,
        "T31",
        "Requirement 2: Transcription and structured transcript",
        bool(good_calls),
        "",
        {"calls_checked": len(good_calls)},
    )
    mark_requirement(
        results,
        "T32",
        "Requirement 3: Intent, confidence, priority, routing and entities",
        bool(good_calls)
        and all(item["intent_id"] and item["suggested_group"] for item in good_calls),
        "",
        {"sample_intent": good_calls[0]["intent_id"] if good_calls else ""},
    )
    mark_requirement(
        results,
        "T33",
        "Requirement 4: Ticket registration in external system",
        get_result_status(results, "T20") == "passed",
        "need at least one completed call with created ticket" if get_result_status(results, "T20") != "passed" else "",
    )
    mark_requirement(
        results,
        "T34",
        "Requirement 5: Notification handoff to responsible side",
        get_result_status(results, "T25") == "passed",
        "current architecture delegates notifications to the linked ticket system"
        if get_result_status(results, "T25") != "passed"
        else "",
    )
    mark_requirement(
        results,
        "T35",
        "Detailed requirements: Storage and audit",
        get_result_status(results, "T21") == "passed"
        and get_result_status(results, "T22") == "passed"
        and isinstance(queue_snapshot, dict)
        and isinstance(audit_snapshot, dict)
        and isinstance((audit_snapshot or {}).get("details"), dict),
        "",
    )
    mark_requirement(
        results,
        "T36",
        "Detailed requirements: Security and role model",
        get_result_status(results, "T1") == "passed"
        and get_result_status(results, "T3") == "passed"
        and get_result_status(results, "T23") == "passed"
        and get_result_status(results, "T24") == "passed",
        "",
    )

    summary = {
        "total": len(results),
        "passed": sum(1 for x in results if x["status"] == "passed"),
        "failed": sum(1 for x in results if x["status"] == "failed"),
        "skipped": sum(1 for x in results if x["status"] == "skipped"),
    }
    requirements = [
        x
        for x in results
        if x["id"] in {"T30", "T31", "T32", "T33", "T34", "T35", "T36"}
    ]
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "base_url": base_url,
        "audio_files": [str(x) for x in audio_files],
        "summary": summary,
        "requirements": requirements,
        "results": results,
    }

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (Path.cwd() / "functional_test_report.json")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] JSON report: {out_json}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
