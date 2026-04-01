#!/usr/bin/env python3
"""Benchmark local Ollama models on the production ticket summary prompt.

The script reads the live prompt template from
services/ticket_creation/internal/services/prompts.go, runs one or more
installed Ollama models on small / medium / large benchmark cases, and reports
both speed and a simple completeness heuristic.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CASE_FILE = REPO_ROOT / "scripts" / "fixtures" / "ollama_summary_cases.json"
PROD_PROMPT_FILE = (
    REPO_ROOT / "services" / "ticket_creation" / "internal" / "services" / "prompts.go"
)


@dataclass
class BenchmarkCase:
    case_id: str
    size: str
    label: str
    transcript: str
    intent_id: str
    priority: str
    entities_info: str
    expected_keyword_groups: list[list[str]]
    source_path: str | None = None


@dataclass
class PromptTemplates:
    system_prompt: str
    user_prompt_template: str


@dataclass
class RunResult:
    model: str
    case_id: str
    case_size: str
    case_label: str
    run_index: int
    success: bool
    http_status: int | None
    transcript_chars: int
    prompt_chars: int
    wall_clock_sec: float | None
    total_duration_sec: float | None
    prompt_eval_sec: float | None
    eval_sec: float | None
    load_sec: float | None
    prompt_eval_count: int | None
    eval_count: int | None
    eval_tokens_per_sec: float | None
    response_chars: int | None
    valid_json: bool
    non_empty_problem: bool
    single_sentence: bool
    keyword_group_hits: int
    keyword_group_total: int
    keyword_coverage: float | None
    completeness_score: float | None
    problem: str | None
    raw_response: str | None
    error: str | None


def normalize_text(value: str) -> str:
    lowered = value.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", re.sub(r"[^a-zа-я0-9\s]+", " ", lowered)).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local Ollama models on the production ticket summary prompt."
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Ollama model name. Pass multiple times for comparison.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print locally available Ollama models and exit.",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Local Ollama base URL.",
    )
    parser.add_argument(
        "--case-file",
        dest="case_files",
        action="append",
        help="JSON file with benchmark cases. May be passed multiple times.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="Benchmark case id to run. May be passed multiple times. Defaults to all cases.",
    )
    parser.add_argument(
        "--transcript-file",
        dest="transcript_files",
        action="append",
        help="Path to a .txt transcript file. May be passed multiple times.",
    )
    parser.add_argument(
        "--transcript-dir",
        dest="transcript_dirs",
        action="append",
        help="Directory with .txt transcript files. Scanned recursively.",
    )
    parser.add_argument(
        "--intent-id",
        default="general_request",
        help="Default intent id for txt cases without sidecar metadata.",
    )
    parser.add_argument(
        "--priority",
        default="medium",
        help="Default priority for txt cases without sidecar metadata.",
    )
    parser.add_argument(
        "--entities-info",
        default="Не извлечены",
        help="Default entities block for txt cases without sidecar metadata.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="How many times to run each model/case pair.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Ollama temperature.",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=160,
        help="Ollama num_predict.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Optional Ollama num_ctx.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one warm-up call per model before measuring.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save raw benchmark results.",
    )
    parser.add_argument(
        "--markdown-report",
        help="Optional path to save a Markdown benchmark report.",
    )
    parser.add_argument(
        "--report-title",
        default="Ollama Summary Benchmark Report",
        help="Title for the optional Markdown report.",
    )
    return parser.parse_args()


def load_prompt_templates(path: Path) -> PromptTemplates:
    source = path.read_text(encoding="utf-8")

    system_match = re.search(
        r"func ticketSummarySystemPrompt\(\) string \{\s*return `(.*?)`\s*\}",
        source,
        flags=re.DOTALL,
    )
    user_match = re.search(
        r"func buildSummaryUserPrompt\(.*?\) string \{\s*return fmt\.Sprintf\(`(.*?)`, transcript, intentID, priority, entitiesInfo\)\s*\}",
        source,
        flags=re.DOTALL,
    )

    if not system_match or not user_match:
        raise ValueError(f"Failed to extract production prompts from {path}")

    return PromptTemplates(
        system_prompt=system_match.group(1),
        user_prompt_template=user_match.group(1),
    )


def load_cases(path: Path, selected_ids: set[str] | None) -> list[BenchmarkCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in payload:
        case = BenchmarkCase(
            case_id=str(item["case_id"]),
            size=str(item["size"]),
            label=str(item["label"]),
            transcript=str(item["transcript"]).strip(),
            intent_id=str(item["intent_id"]),
            priority=str(item["priority"]),
            entities_info=str(item.get("entities_info", "Не извлечены")).strip(),
            expected_keyword_groups=[
                [str(keyword).strip() for keyword in group if str(keyword).strip()]
                for group in item.get("expected_keyword_groups", [])
            ],
            source_path=str(path),
        )
        if selected_ids and case.case_id not in selected_ids:
            continue
        cases.append(case)

    if selected_ids:
        found = {case.case_id for case in cases}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(missing)}")

    if not cases:
        raise ValueError("No benchmark cases loaded")

    return cases


def discover_transcript_files(
    transcript_files: list[str] | None,
    transcript_dirs: list[str] | None,
) -> list[Path]:
    paths: list[Path] = []

    for raw in transcript_files or []:
        path = Path(raw).expanduser().resolve()
        if path.suffix.lower() != ".txt":
            raise ValueError(f"Transcript file must be .txt: {path}")
        if not path.exists():
            raise ValueError(f"Transcript file not found: {path}")
        paths.append(path)

    for raw in transcript_dirs or []:
        directory = Path(raw).expanduser().resolve()
        if not directory.exists():
            raise ValueError(f"Transcript directory not found: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Transcript path is not a directory: {directory}")
        paths.extend(sorted(directory.rglob("*.txt")))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def infer_case_size(path: Path, transcript: str) -> str:
    name = path.stem.lower()
    for size in ("small", "medium", "large"):
        if size in name:
            return size

    length = len(transcript)
    if length <= 500:
        return "small"
    if length <= 1500:
        return "medium"
    return "large"


def load_sidecar_metadata(path: Path) -> dict[str, Any]:
    candidates = [
        path.with_suffix(".meta.json"),
        path.with_name(path.name + ".meta.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Sidecar metadata must be an object: {candidate}")
            return payload
    return {}


def load_txt_cases(
    transcript_paths: list[Path],
    selected_ids: set[str] | None,
    default_intent_id: str,
    default_priority: str,
    default_entities_info: str,
) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    for path in transcript_paths:
        transcript = path.read_text(encoding="utf-8").strip()
        if not transcript:
            raise ValueError(f"Transcript file is empty: {path}")

        meta = load_sidecar_metadata(path)
        case_id = str(meta.get("case_id") or path.stem)
        if selected_ids and case_id not in selected_ids:
            continue

        expected_keyword_groups = [
            [str(keyword).strip() for keyword in group if str(keyword).strip()]
            for group in meta.get("expected_keyword_groups", [])
        ]

        cases.append(
            BenchmarkCase(
                case_id=case_id,
                size=str(meta.get("size") or infer_case_size(path, transcript)),
                label=str(meta.get("label") or path.stem),
                transcript=transcript,
                intent_id=str(meta.get("intent_id") or default_intent_id),
                priority=str(meta.get("priority") or default_priority),
                entities_info=str(meta.get("entities_info") or default_entities_info).strip(),
                expected_keyword_groups=expected_keyword_groups,
                source_path=str(path),
            )
        )

    return cases


def build_cases(args: argparse.Namespace) -> list[BenchmarkCase]:
    selected_ids = set(args.case_ids) if args.case_ids else None
    cases: list[BenchmarkCase] = []

    transcript_paths = discover_transcript_files(args.transcript_files, args.transcript_dirs)
    if transcript_paths:
        cases.extend(
            load_txt_cases(
                transcript_paths=transcript_paths,
                selected_ids=selected_ids,
                default_intent_id=args.intent_id,
                default_priority=args.priority,
                default_entities_info=args.entities_info,
            )
        )

    case_files = [Path(path).expanduser().resolve() for path in args.case_files or []]
    if not case_files and not transcript_paths:
        case_files = [DEFAULT_CASE_FILE]

    for case_file in case_files:
        if not case_file.exists():
            raise ValueError(f"Case file not found: {case_file}")
        cases.extend(load_cases(case_file, selected_ids))

    if selected_ids:
        found = {case.case_id for case in cases}
        missing = sorted(selected_ids - found)
        if missing:
            raise ValueError(f"Unknown case ids: {', '.join(missing)}")

    if not cases:
        raise ValueError("No benchmark cases loaded")

    unique: dict[tuple[str, str], BenchmarkCase] = {}
    for case in cases:
        key = (case.case_id, case.source_path or "")
        unique[key] = case
    return list(unique.values())


def list_models(base_url: str, timeout: int) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"Content-Type": "application/json"},
        method="GET",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [str(item["name"]) for item in payload.get("models", []) if item.get("name")]


def build_user_prompt(template: str, case: BenchmarkCase) -> str:
    return template % (
        case.transcript,
        case.intent_id,
        case.priority,
        case.entities_info or "Не извлечены",
    )


def _ns_to_sec(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000_000.0, 3)
    except (TypeError, ValueError):
        return None


def compute_eval_tokens_per_sec(eval_count: Any, eval_duration_ns: Any) -> float | None:
    try:
        if eval_count is None or eval_duration_ns in (None, 0):
            return None
        seconds = float(eval_duration_ns) / 1_000_000_000.0
        if seconds <= 0:
            return None
        return round(float(eval_count) / seconds, 2)
    except (TypeError, ValueError):
        return None


def count_sentences(text: str) -> int:
    if not text.strip():
        return 0
    matches = re.findall(r"[.!?]+", text)
    return max(1, len(matches))


def compute_completeness(
    *,
    valid_json: bool,
    non_empty_problem: bool,
    single_sentence: bool,
    keyword_coverage: float | None,
) -> float:
    weights: list[tuple[bool, float]] = [
        (valid_json, 0.2),
        (non_empty_problem, 0.45),
        (single_sentence, 0.15),
    ]

    score = sum(weight for ok, weight in weights if ok)
    total_weight = sum(weight for _, weight in weights)

    if keyword_coverage is not None:
        score += 0.2 * keyword_coverage
        total_weight += 0.2

    if total_weight <= 0:
        return 0.0
    return round(score / total_weight, 3)


def evaluate_problem(problem: str, case: BenchmarkCase) -> tuple[bool, bool, int, int, float | None, float]:
    normalized_problem = normalize_text(problem)
    non_empty_problem = bool(normalized_problem)
    single_sentence = count_sentences(problem) <= 1

    hits = 0
    total = len(case.expected_keyword_groups)
    for group in case.expected_keyword_groups:
        if any(normalize_text(keyword) in normalized_problem for keyword in group):
            hits += 1

    keyword_coverage = round(hits / total, 3) if total else None
    completeness = compute_completeness(
        valid_json=True,
        non_empty_problem=non_empty_problem,
        single_sentence=single_sentence,
        keyword_coverage=keyword_coverage,
    )

    return non_empty_problem, single_sentence, hits, total, keyword_coverage, completeness


def _parse_problem_from_response(raw_response: str) -> tuple[bool, str]:
    candidate = raw_response.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]

    payload = json.loads(candidate)
    return True, str(payload.get("problem", "")).strip()


def _make_request(
    base_url: str,
    model: str,
    system_prompt: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    num_ctx: int | None,
    timeout: int,
) -> tuple[int, dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }
    if num_ctx:
        payload["options"]["num_ctx"] = num_ctx

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    started = time.perf_counter()
    with opener.open(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        status = response.status
    elapsed = time.perf_counter() - started
    return status, json.loads(body), elapsed


def benchmark_one(
    base_url: str,
    model: str,
    templates: PromptTemplates,
    case: BenchmarkCase,
    temperature: float,
    num_predict: int,
    num_ctx: int | None,
    timeout: int,
    run_index: int,
) -> RunResult:
    prompt = build_user_prompt(templates.user_prompt_template, case)

    try:
        http_status, payload, wall_clock = _make_request(
            base_url=base_url,
            model=model,
            system_prompt=templates.system_prompt,
            prompt=prompt,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            timeout=timeout,
        )
        raw_response = str(payload.get("response", ""))
        valid_json, problem = _parse_problem_from_response(raw_response)
        non_empty_problem, single_sentence, hits, total, keyword_coverage, completeness = evaluate_problem(
            problem,
            case,
        )

        return RunResult(
            model=model,
            case_id=case.case_id,
            case_size=case.size,
            case_label=case.label,
            run_index=run_index,
            success=True,
            http_status=http_status,
            transcript_chars=len(case.transcript),
            prompt_chars=len(prompt),
            wall_clock_sec=round(wall_clock, 3),
            total_duration_sec=_ns_to_sec(payload.get("total_duration")),
            prompt_eval_sec=_ns_to_sec(payload.get("prompt_eval_duration")),
            eval_sec=_ns_to_sec(payload.get("eval_duration")),
            load_sec=_ns_to_sec(payload.get("load_duration")),
            prompt_eval_count=payload.get("prompt_eval_count"),
            eval_count=payload.get("eval_count"),
            eval_tokens_per_sec=compute_eval_tokens_per_sec(
                payload.get("eval_count"),
                payload.get("eval_duration"),
            ),
            response_chars=len(raw_response),
            valid_json=valid_json,
            non_empty_problem=non_empty_problem,
            single_sentence=single_sentence,
            keyword_group_hits=hits,
            keyword_group_total=total,
            keyword_coverage=keyword_coverage,
            completeness_score=completeness,
            problem=problem,
            raw_response=raw_response,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return RunResult(
            model=model,
            case_id=case.case_id,
            case_size=case.size,
            case_label=case.label,
            run_index=run_index,
            success=False,
            http_status=exc.code,
            transcript_chars=len(case.transcript),
            prompt_chars=len(prompt),
            wall_clock_sec=None,
            total_duration_sec=None,
            prompt_eval_sec=None,
            eval_sec=None,
            load_sec=None,
            prompt_eval_count=None,
            eval_count=None,
            eval_tokens_per_sec=None,
            response_chars=None,
            valid_json=False,
            non_empty_problem=False,
            single_sentence=False,
            keyword_group_hits=0,
            keyword_group_total=len(case.expected_keyword_groups),
            keyword_coverage=0.0 if case.expected_keyword_groups else None,
            completeness_score=0.0,
            problem=None,
            raw_response=None,
            error=f"HTTP {exc.code}: {body.strip()}",
        )
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            model=model,
            case_id=case.case_id,
            case_size=case.size,
            case_label=case.label,
            run_index=run_index,
            success=False,
            http_status=None,
            transcript_chars=len(case.transcript),
            prompt_chars=len(prompt),
            wall_clock_sec=None,
            total_duration_sec=None,
            prompt_eval_sec=None,
            eval_sec=None,
            load_sec=None,
            prompt_eval_count=None,
            eval_count=None,
            eval_tokens_per_sec=None,
            response_chars=None,
            valid_json=False,
            non_empty_problem=False,
            single_sentence=False,
            keyword_group_hits=0,
            keyword_group_total=len(case.expected_keyword_groups),
            keyword_coverage=0.0 if case.expected_keyword_groups else None,
            completeness_score=0.0,
            problem=None,
            raw_response=None,
            error=str(exc),
        )


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return round(statistics.mean(clean), 3)


def _format_float(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def summarize_by_model(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

    rows: list[dict[str, Any]] = []
    for model, items in sorted(grouped.items()):
        ok_count = sum(1 for item in items if item.success)
        json_ok = sum(1 for item in items if item.valid_json)
        problem_ok = sum(1 for item in items if item.non_empty_problem)
        rows.append(
            {
                "model": model,
                "runs": len(items),
                "ok": ok_count,
                "wall_avg_s": _safe_mean([item.wall_clock_sec for item in items]),
                "tok_s_avg": _safe_mean([item.eval_tokens_per_sec for item in items]),
                "complete_avg": _safe_mean([item.completeness_score for item in items]),
                "json_ok": f"{json_ok}/{len(items)}",
                "problem_ok": f"{problem_ok}/{len(items)}",
            }
        )
    return rows


def print_summary_by_model(results: list[RunResult]) -> None:
    rows = summarize_by_model(results)
    print("\nOverall summary by model:")
    print(
        "model".ljust(20),
        "runs".rjust(4),
        "ok".rjust(4),
        "wall_avg_s".rjust(12),
        "tok_s_avg".rjust(11),
        "complete_avg".rjust(13),
        "json_ok".rjust(8),
        "problem_ok".rjust(11),
    )
    for row in rows:
        print(
            row["model"].ljust(20),
            str(row["runs"]).rjust(4),
            str(row["ok"]).rjust(4),
            _format_float(row["wall_avg_s"]).rjust(12),
            _format_float(row["tok_s_avg"], 2).rjust(11),
            _format_float(row["complete_avg"]).rjust(13),
            str(row["json_ok"]).rjust(8),
            str(row["problem_ok"]).rjust(11),
        )


def summarize_by_case(results: list[RunResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.model, result.case_id), []).append(result)

    rows: list[dict[str, Any]] = []
    for (model, case_id), items in sorted(grouped.items()):
        size = items[0].case_size
        rows.append(
            {
                "model": model,
                "case": case_id,
                "size": size,
                "wall_avg_s": _safe_mean([item.wall_clock_sec for item in items]),
                "tok_s_avg": _safe_mean([item.eval_tokens_per_sec for item in items]),
                "coverage": _safe_mean([item.keyword_coverage for item in items]),
                "complete": _safe_mean([item.completeness_score for item in items]),
            }
        )
    return rows


def print_summary_by_case(results: list[RunResult]) -> None:
    rows = summarize_by_case(results)
    print("\nSummary by model and case:")
    print(
        "model".ljust(20),
        "case".ljust(10),
        "size".ljust(8),
        "wall_avg_s".rjust(12),
        "tok_s_avg".rjust(11),
        "coverage".rjust(10),
        "complete".rjust(10),
    )
    for row in rows:
        print(
            row["model"].ljust(20),
            row["case"].ljust(10),
            row["size"].ljust(8),
            _format_float(row["wall_avg_s"]).rjust(12),
            _format_float(row["tok_s_avg"], 2).rjust(11),
            _format_float(row["coverage"]).rjust(10),
            _format_float(row["complete"]).rjust(10),
        )


def print_case_outputs(results: list[RunResult]) -> None:
    print("\nLatest model outputs by case:")
    latest: dict[tuple[str, str], RunResult] = {}
    for item in results:
        latest[(item.model, item.case_id)] = item

    for (model, case_id), item in sorted(latest.items()):
        print(
            f"- model={model} case={case_id} size={item.case_size} "
            f"wall={item.wall_clock_sec} completeness={item.completeness_score}"
        )
        if item.problem:
            print(f"  problem={item.problem}")
        if item.error:
            print(f"  error={item.error}")


def latest_case_outputs(results: list[RunResult]) -> list[RunResult]:
    latest: dict[tuple[str, str], RunResult] = {}
    for item in results:
        latest[(item.model, item.case_id)] = item
    return [latest[key] for key in sorted(latest.keys())]


def escape_md_cell(value: Any) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def build_markdown_report(
    args: argparse.Namespace,
    cases: list[BenchmarkCase],
    results: list[RunResult],
) -> str:
    report_time = datetime.now().astimezone().isoformat(timespec="seconds")
    model_rows = summarize_by_model(results)
    case_rows = summarize_by_case(results)
    latest_rows = latest_case_outputs(results)

    lines: list[str] = [
        f"# {args.report_title}",
        "",
        f"_Generated: {report_time}_",
        "",
        "## Benchmark Setup",
        "",
        f"- Production prompt source: `{PROD_PROMPT_FILE}`",
        f"- Benchmark script: `{Path(__file__).resolve()}`",
        f"- Ollama base URL: `{args.base_url}`",
        f"- Models: `{', '.join(args.models or [])}`",
        f"- Runs per case: `{args.runs}`",
        f"- Temperature: `{args.temperature}`",
        f"- Num predict: `{args.num_predict}`",
        f"- Num ctx: `{args.num_ctx if args.num_ctx is not None else 'default'}`",
        "",
        "## Cases",
        "",
        "| Case | Size | Intent | Priority | Source |",
        "|---|---|---|---|---|",
    ]

    for case in cases:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md_cell(case.case_id),
                    escape_md_cell(case.size),
                    escape_md_cell(case.intent_id),
                    escape_md_cell(case.priority),
                    escape_md_cell(case.source_path or "inline"),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Overall Summary By Model",
            "",
            "| Model | Runs | OK | Avg wall time, s | Avg tokens/s | Avg completeness | JSON OK | Problem OK |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for row in model_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md_cell(row["model"]),
                    str(row["runs"]),
                    str(row["ok"]),
                    _format_float(row["wall_avg_s"]),
                    _format_float(row["tok_s_avg"], 2),
                    _format_float(row["complete_avg"]),
                    escape_md_cell(row["json_ok"]),
                    escape_md_cell(row["problem_ok"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Summary By Model And Case",
            "",
            "| Model | Case | Size | Avg wall time, s | Avg tokens/s | Coverage | Completeness |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )

    for row in case_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_md_cell(row["model"]),
                    escape_md_cell(row["case"]),
                    escape_md_cell(row["size"]),
                    _format_float(row["wall_avg_s"]),
                    _format_float(row["tok_s_avg"], 2),
                    _format_float(row["coverage"]),
                    _format_float(row["complete"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Representative Outputs",
            "",
        ]
    )

    for item in latest_rows:
        lines.append(
            f"- `{item.model}` / `{item.case_id}` / `{item.case_size}`: "
            f"wall=`{item.wall_clock_sec}` completeness=`{item.completeness_score}`"
        )
        if item.problem:
            lines.append(f"  Output: `{item.problem}`")
        if item.error:
            lines.append(f"  Error: `{item.error}`")

    lines.extend(
        [
            "",
            "## Completeness Scoring",
            "",
            "- `20%`: valid JSON",
            "- `45%`: non-empty `problem` field",
            "- `15%`: single-sentence response",
            "- `20%`: expected keyword-group coverage when sidecar metadata is provided",
        ]
    )

    return "\n".join(lines) + "\n"


def warmup_model(
    base_url: str,
    model: str,
    templates: PromptTemplates,
    case: BenchmarkCase,
    temperature: float,
    num_predict: int,
    num_ctx: int | None,
    timeout: int,
) -> None:
    prompt = build_user_prompt(templates.user_prompt_template, case)
    _make_request(
        base_url=base_url,
        model=model,
        system_prompt=templates.system_prompt,
        prompt=prompt,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
        timeout=timeout,
    )


def main() -> int:
    args = parse_args()

    try:
        templates = load_prompt_templates(PROD_PROMPT_FILE)
        cases = build_cases(args)
        available_models = list_models(args.base_url, args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

    if args.list_models:
        if not available_models:
            print("No local Ollama models found.")
            return 0
        print("Local Ollama models:")
        for model in available_models:
            print(f"- {model}")
        return 0

    models = args.models or available_models
    if not models:
        print("No models specified and no local Ollama models were found.", file=sys.stderr)
        return 1

    unknown_models = sorted(set(models) - set(available_models))
    if unknown_models:
        print(
            "These models are not available locally:",
            *unknown_models,
            sep="\n- ",
            file=sys.stderr,
        )
        return 1

    print("Using production prompts from:", PROD_PROMPT_FILE)
    if args.case_files:
        print("Using JSON case files:", ", ".join(str(Path(path).expanduser().resolve()) for path in args.case_files))
    if args.transcript_files or args.transcript_dirs:
        print("Using txt transcript inputs")
    print("Models:", ", ".join(models))

    if args.warmup:
        warmup_case = cases[0]
        for model in models:
            print(f"Warming up model={model} with case={warmup_case.case_id}...")
            warmup_model(
                base_url=args.base_url,
                model=model,
                templates=templates,
                case=warmup_case,
                temperature=args.temperature,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
                timeout=args.timeout,
            )

    results: list[RunResult] = []
    for case in cases:
        for model in models:
            for run_index in range(1, args.runs + 1):
                print(
                    f"Running model={model} case={case.case_id} "
                    f"size={case.size} run={run_index}/{args.runs}..."
                )
                results.append(
                    benchmark_one(
                        base_url=args.base_url,
                        model=model,
                        templates=templates,
                        case=case,
                        temperature=args.temperature,
                        num_predict=args.num_predict,
                        num_ctx=args.num_ctx,
                        timeout=args.timeout,
                        run_index=run_index,
                    )
                )

    print_summary_by_model(results)
    print_summary_by_case(results)
    print_case_outputs(results)

    print("\nCompleteness scoring note:")
    print("- 20%: valid JSON")
    print("- 45%: non-empty problem")
    print("- 15%: one-sentence response")
    print("- 20%: expected keyword-group coverage when sidecar keywords are provided")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved raw results to: {output_path}")

    if args.markdown_report:
        report_path = Path(args.markdown_report).expanduser().resolve()
        report_path.write_text(
            build_markdown_report(args, cases, results),
            encoding="utf-8",
        )
        print(f"Saved Markdown report to: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
