#!/usr/bin/env python3
"""Benchmark Ollama summary generation for ticket_creation prompts.

This script reproduces the same high-level prompt shape used by
services/ticket_creation/internal/services/summarizer.go and compares one or
more Ollama models on the same transcript(s).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """Ты формируешь стандартизированное описание тикета по транскрипции телефонного звонка.
Не генерируй заголовок, тип обращения, приоритет, теги, блоки "контекст" или "действие", а также пояснения вне JSON.
Верни только JSON-объект без markdown и без дополнительного текста.

Формат ответа:
{
  "problem": "1 короткое предложение с сутью проблемы клиента"
}

Правила:
- Пиши по-русски.
- Поле problem: максимум одно короткое предложение.
- Не выдумывай факты, которых нет в транскрипции.
- Не повторяй intent, это поле уже известно системе.
- Если данных недостаточно, верни пустую строку.
- Не включай персональные данные без необходимости."""


def build_user_prompt(transcript: str, intent_id: str, priority: str, entities_info: str) -> str:
    return f"""Транскрипция звонка:
{transcript}

Тип обращения (не дублируй в ответе): {intent_id}
Приоритет: {priority}

Извлечённые сущности:
{entities_info}

Сформируй краткую формулировку проблемы по заданному JSON-формату."""


@dataclass
class RunResult:
    model: str
    transcript_file: str
    run_index: int
    success: bool
    http_status: int | None
    wall_clock_sec: float | None
    total_duration_sec: float | None
    prompt_eval_sec: float | None
    eval_sec: float | None
    load_sec: float | None
    prompt_eval_count: int | None
    eval_count: int | None
    response_chars: int | None
    problem: str | None
    error: str | None


def _ns_to_sec(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value) / 1_000_000_000.0, 3)
    except (TypeError, ValueError):
        return None


def _parse_nested_problem(raw_response: str) -> str:
    candidate = raw_response.strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    payload = json.loads(candidate)
    return str(payload.get("problem", "")).strip()


def _make_request(
    base_url: str,
    model: str,
    prompt: str,
    temperature: float,
    num_predict: int,
    num_ctx: int | None,
    timeout: int,
) -> tuple[int, dict[str, Any], float]:
    payload: dict[str, Any] = {
        "model": model,
        "system": SYSTEM_PROMPT,
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

    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        status = response.status
    elapsed = time.perf_counter() - started
    return status, json.loads(body), elapsed


def benchmark_one(
    base_url: str,
    model: str,
    transcript_file: Path,
    intent_id: str,
    priority: str,
    entities_info: str,
    temperature: float,
    num_predict: int,
    num_ctx: int | None,
    timeout: int,
    run_index: int,
) -> RunResult:
    transcript = transcript_file.read_text(encoding="utf-8")
    prompt = build_user_prompt(transcript, intent_id, priority, entities_info)

    try:
        http_status, payload, wall_clock = _make_request(
            base_url=base_url,
            model=model,
            prompt=prompt,
            temperature=temperature,
            num_predict=num_predict,
            num_ctx=num_ctx,
            timeout=timeout,
        )
        raw_response = str(payload.get("response", ""))
        problem = _parse_nested_problem(raw_response)
        return RunResult(
            model=model,
            transcript_file=str(transcript_file),
            run_index=run_index,
            success=True,
            http_status=http_status,
            wall_clock_sec=round(wall_clock, 3),
            total_duration_sec=_ns_to_sec(payload.get("total_duration")),
            prompt_eval_sec=_ns_to_sec(payload.get("prompt_eval_duration")),
            eval_sec=_ns_to_sec(payload.get("eval_duration")),
            load_sec=_ns_to_sec(payload.get("load_duration")),
            prompt_eval_count=payload.get("prompt_eval_count"),
            eval_count=payload.get("eval_count"),
            response_chars=len(raw_response),
            problem=problem,
            error=None,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return RunResult(
            model=model,
            transcript_file=str(transcript_file),
            run_index=run_index,
            success=False,
            http_status=exc.code,
            wall_clock_sec=None,
            total_duration_sec=None,
            prompt_eval_sec=None,
            eval_sec=None,
            load_sec=None,
            prompt_eval_count=None,
            eval_count=None,
            response_chars=None,
            problem=None,
            error=f"HTTP {exc.code}: {body.strip()}",
        )
    except Exception as exc:  # noqa: BLE001
        return RunResult(
            model=model,
            transcript_file=str(transcript_file),
            run_index=run_index,
            success=False,
            http_status=None,
            wall_clock_sec=None,
            total_duration_sec=None,
            prompt_eval_sec=None,
            eval_sec=None,
            load_sec=None,
            prompt_eval_count=None,
            eval_count=None,
            response_chars=None,
            problem=None,
            error=str(exc),
        )


def _safe_mean(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.mean(clean), 3)


def _format_float(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_summary(results: list[RunResult]) -> None:
    grouped: dict[str, list[RunResult]] = {}
    for result in results:
        grouped.setdefault(result.model, []).append(result)

    print("\nSummary by model:")
    print(
        "model".ljust(18),
        "runs".rjust(4),
        "ok".rjust(4),
        "wall_avg_s".rjust(12),
        "total_avg_s".rjust(12),
        "prompt_avg_s".rjust(13),
        "eval_avg_s".rjust(11),
        "prompt_tok_avg".rjust(15),
    )
    for model, items in grouped.items():
        ok_count = sum(1 for item in items if item.success)
        print(
            model.ljust(18),
            str(len(items)).rjust(4),
            str(ok_count).rjust(4),
            _format_float(_safe_mean([item.wall_clock_sec for item in items])).rjust(12),
            _format_float(_safe_mean([item.total_duration_sec for item in items])).rjust(12),
            _format_float(_safe_mean([item.prompt_eval_sec for item in items])).rjust(13),
            _format_float(_safe_mean([item.eval_sec for item in items])).rjust(11),
            _format_float(_safe_mean([float(item.prompt_eval_count) if item.prompt_eval_count is not None else None for item in items])).rjust(15),
        )


def print_details(results: list[RunResult]) -> None:
    print("\nDetailed runs:")
    for item in results:
        print(
            f"- model={item.model} file={item.transcript_file} run={item.run_index} "
            f"success={item.success} wall={item.wall_clock_sec} total={item.total_duration_sec} "
            f"prompt_eval={item.prompt_eval_sec} eval={item.eval_sec}"
        )
        if item.problem:
            print(f"  problem={item.problem}")
        if item.error:
            print(f"  error={item.error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Ollama summary generation on transcript files.")
    parser.add_argument(
        "--transcript-file",
        dest="transcript_files",
        action="append",
        required=True,
        help="Path to plain-text transcript file. May be passed multiple times.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        required=True,
        help="Ollama model name. May be passed multiple times for A/B comparison.",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Ollama base URL.")
    parser.add_argument("--intent-id", default="consulting", help="Intent id used in the prompt.")
    parser.add_argument("--priority", default="medium", help="Priority used in the prompt.")
    parser.add_argument("--entities-info", default="Не извлечены", help="Entities block used in the prompt.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Ollama temperature.")
    parser.add_argument("--num-predict", type=int, default=80, help="Ollama num_predict.")
    parser.add_argument("--num-ctx", type=int, default=None, help="Optional Ollama num_ctx.")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout in seconds.")
    parser.add_argument("--runs", type=int, default=1, help="How many times to run each model/file pair.")
    parser.add_argument("--output-json", help="Optional path to save raw benchmark results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    transcript_files = [Path(path).expanduser().resolve() for path in args.transcript_files]
    missing = [str(path) for path in transcript_files if not path.exists()]
    if missing:
        print("Missing transcript files:", *missing, sep="\n- ", file=sys.stderr)
        return 1

    results: list[RunResult] = []
    for transcript_file in transcript_files:
        for model in args.models:
            for run_index in range(1, args.runs + 1):
                print(f"Running model={model} file={transcript_file} run={run_index}/{args.runs}...")
                result = benchmark_one(
                    base_url=args.base_url,
                    model=model,
                    transcript_file=transcript_file,
                    intent_id=args.intent_id,
                    priority=args.priority,
                    entities_info=args.entities_info,
                    temperature=args.temperature,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                    timeout=args.timeout,
                    run_index=run_index,
                )
                results.append(result)

    print_summary(results)
    print_details(results)

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
