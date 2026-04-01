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
        default=str(DEFAULT_CASE_FILE),
        help="JSON file with benchmark cases.",
    )
    parser.add_argument(
        "--case",
        dest="case_ids",
        action="append",
        help="Benchmark case id to run. May be passed multiple times. Defaults to all cases.",
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


def evaluate_problem(problem: str, case: BenchmarkCase) -> tuple[bool, int, int, float | None, float]:
    normalized_problem = normalize_text(problem)
    non_empty_problem = bool(normalized_problem)
    single_sentence = count_sentences(problem) <= 1

    hits = 0
    total = len(case.expected_keyword_groups)
    for group in case.expected_keyword_groups:
        if any(normalize_text(keyword) in normalized_problem for keyword in group):
            hits += 1

    keyword_coverage = round(hits / total, 3) if total else None
    completeness = 0.0
    completeness += 0.25 if non_empty_problem else 0.0
    completeness += 0.15 if single_sentence else 0.0
    completeness += 0.60 * (keyword_coverage or 0.0)

    return single_sentence, hits, total, keyword_coverage, round(completeness, 3)


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
        single_sentence, hits, total, keyword_coverage, completeness = evaluate_problem(problem, case)

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
            non_empty_problem=bool(problem.strip()),
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


def print_summary_by_model(results: list[RunResult]) -> None:
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

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
    for model, items in grouped.items():
        ok_count = sum(1 for item in items if item.success)
        json_ok = sum(1 for item in items if item.valid_json)
        problem_ok = sum(1 for item in items if item.non_empty_problem)
        print(
            model.ljust(20),
            str(len(items)).rjust(4),
            str(ok_count).rjust(4),
            _format_float(_safe_mean([item.wall_clock_sec for item in items])).rjust(12),
            _format_float(_safe_mean([item.eval_tokens_per_sec for item in items]), 2).rjust(11),
            _format_float(_safe_mean([item.completeness_score for item in items])).rjust(13),
            f"{json_ok}/{len(items)}".rjust(8),
            f"{problem_ok}/{len(items)}".rjust(11),
        )


def print_summary_by_case(results: list[RunResult]) -> None:
    grouped: dict[tuple[str, str], list[RunResult]] = {}
    for result in results:
        grouped.setdefault((result.model, result.case_id), []).append(result)

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
    for (model, case_id), items in sorted(grouped.items()):
        size = items[0].case_size
        print(
            model.ljust(20),
            case_id.ljust(10),
            size.ljust(8),
            _format_float(_safe_mean([item.wall_clock_sec for item in items])).rjust(12),
            _format_float(_safe_mean([item.eval_tokens_per_sec for item in items]), 2).rjust(11),
            _format_float(_safe_mean([item.keyword_coverage for item in items])).rjust(10),
            _format_float(_safe_mean([item.completeness_score for item in items])).rjust(10),
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
        cases = load_cases(
            Path(args.case_file).expanduser().resolve(),
            set(args.case_ids) if args.case_ids else None,
        )
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
    print("Using benchmark cases from:", Path(args.case_file).expanduser().resolve())
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
    print("- 25%: non-empty problem")
    print("- 15%: one-sentence response")
    print("- 60%: expected keyword-group coverage for the benchmark case")

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.write_text(
            json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved raw results to: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
