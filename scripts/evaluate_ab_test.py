#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def _read_csv(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers


def _pick_column(headers: Sequence[str], preferred: str, fallbacks: Iterable[str]) -> str:
    if preferred and preferred in headers:
        return preferred
    for name in fallbacks:
        if name in headers:
            return name
    return ""


def _parse_float(value: object) -> Optional[float]:
    raw = str(value or "").strip().replace(",", ".")
    if raw == "":
        return None
    if raw.endswith("%"):
        raw = raw[:-1]
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_boolish(value: object) -> Optional[bool]:
    raw = str(value or "").strip().lower()
    if raw == "":
        return None
    if raw in {"1", "true", "yes", "y", "error", "wrong", "failed"}:
        return True
    if raw in {"0", "false", "no", "n", "ok", "correct", "passed"}:
        return False
    numeric = _parse_float(raw)
    if numeric is not None:
        return numeric > 0
    return None


def _parse_percentage(value: object) -> Optional[float]:
    numeric = _parse_float(value)
    if numeric is None:
        return None
    if 0.0 <= numeric <= 1.0:
        return numeric * 100.0
    return numeric


def _percentile(values: Sequence[float], percentile: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _round_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _reduction(control: Optional[float], treatment: Optional[float]) -> Optional[float]:
    if control is None or treatment is None or control == 0:
        return None
    return round((control - treatment) / control * 100.0, 6)


def _build_group_stats(rows: Sequence[Dict[str, str]], group_col: str, time_col: str, error_col: str, operator_load_col: str) -> Dict[str, Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    for row in rows:
        group_name = str(row.get(group_col) or "").strip()
        if not group_name:
            group_name = "unknown"
        bucket = grouped.setdefault(
            group_name,
            {
                "samples": 0,
                "time_values": [],
                "error_values": [],
                "operator_load_values": [],
            },
        )
        bucket["samples"] += 1

        if time_col:
            time_value = _parse_float(row.get(time_col))
            if time_value is not None:
                bucket["time_values"].append(time_value)

        if error_col:
            error_value = _parse_boolish(row.get(error_col))
            if error_value is not None:
                bucket["error_values"].append(1.0 if error_value else 0.0)

        if operator_load_col:
            load_value = _parse_percentage(row.get(operator_load_col))
            if load_value is not None:
                bucket["operator_load_values"].append(load_value)

    summary: Dict[str, Dict[str, object]] = {}
    for group_name, bucket in grouped.items():
        time_values = [float(v) for v in bucket["time_values"]]
        error_values = [float(v) for v in bucket["error_values"]]
        load_values = [float(v) for v in bucket["operator_load_values"]]
        summary[group_name] = {
            "samples": int(bucket["samples"]),
            "avg_time_sec": _round_or_none(statistics.mean(time_values) if time_values else None),
            "median_time_sec": _round_or_none(statistics.median(time_values) if time_values else None),
            "p95_time_sec": _round_or_none(_percentile(time_values, 0.95)),
            "error_rate": _round_or_none(statistics.mean(error_values) if error_values else None),
            "avg_operator_load_pct": _round_or_none(statistics.mean(load_values) if load_values else None),
            "time_values_count": len(time_values),
            "error_values_count": len(error_values),
            "operator_load_values_count": len(load_values),
        }
    return dict(sorted(summary.items()))


def _print_report(report: Dict[str, object]) -> None:
    print("\n== A/B Test ==")
    print(f"columns: group='{report['group_col']}', time='{report['time_col']}', error='{report['error_col']}', operator_load='{report['operator_load_col']}'")
    for group_name, stats in report["groups"].items():
        print(
            "group={group} samples={samples} avg_time={avg_time} error_rate={error_rate} avg_operator_load={load}".format(
                group=group_name,
                samples=stats["samples"],
                avg_time=stats["avg_time_sec"],
                error_rate=stats["error_rate"],
                load=stats["avg_operator_load_pct"],
            )
        )

    comparison = report.get("comparison") or {}
    if comparison:
        print(
            "comparison: time_reduction={time_red}%, error_reduction={error_red}%, operator_load_reduction={load_red}%".format(
                time_red=comparison.get("time_reduction_pct"),
                error_red=comparison.get("error_reduction_pct"),
                load_red=comparison.get("operator_load_reduction_pct"),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate A/B test results from CSV.")
    parser.add_argument("--csv", required=True, help="Path to CSV with A/B results.")
    parser.add_argument("--group-col", default="group", help="Column with A/B group label.")
    parser.add_argument("--time-col", default="classification_time_sec", help="Column with per-case classification time in seconds.")
    parser.add_argument("--error-col", default="is_error", help="Column with classification error flag.")
    parser.add_argument("--operator-load-col", default="operator_load_pct", help="Column with operator involvement/load in percent.")
    parser.add_argument("--group-a", default="A", help="Control group label.")
    parser.add_argument("--group-b", default="B", help="Treatment group label.")
    parser.add_argument("--out-json", default="", help="Optional path to save JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    rows, headers = _read_csv(csv_path)
    if not rows:
        print(f"[ERROR] csv is empty: {csv_path}")
        return 2

    group_col = _pick_column(headers, args.group_col, ["group", "variant", "ab_group"])
    time_col = _pick_column(headers, args.time_col, ["classification_time_sec", "time_sec", "duration_sec"])
    error_col = _pick_column(headers, args.error_col, ["is_error", "error", "wrong_category"])
    operator_load_col = _pick_column(
        headers,
        args.operator_load_col,
        ["operator_load_pct", "operator_involvement_pct", "operator_share_pct", "operator_load"],
    )
    if not group_col:
        print("[ERROR] group column not found")
        print("headers:", ", ".join(headers))
        return 2

    groups = _build_group_stats(rows, group_col, time_col, error_col, operator_load_col)
    control = groups.get(args.group_a)
    treatment = groups.get(args.group_b)
    comparison: Dict[str, object] = {}
    if control and treatment:
        comparison = {
            "control_group": args.group_a,
            "treatment_group": args.group_b,
            "time_reduction_pct": _reduction(
                control.get("avg_time_sec"), treatment.get("avg_time_sec")
            ),
            "error_reduction_pct": _reduction(
                control.get("error_rate"), treatment.get("error_rate")
            ),
            "operator_load_reduction_pct": _reduction(
                control.get("avg_operator_load_pct"), treatment.get("avg_operator_load_pct")
            ),
        }

    report = {
        "csv": str(csv_path),
        "rows_total": len(rows),
        "group_col": group_col,
        "time_col": time_col,
        "error_col": error_col,
        "operator_load_col": operator_load_col,
        "groups": groups,
        "comparison": comparison,
    }

    _print_report(report)

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (csv_path.parent / "ab_test_metrics.json")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] report saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
