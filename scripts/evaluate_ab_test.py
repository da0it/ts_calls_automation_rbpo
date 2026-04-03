#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path


EMPTY = "__empty__"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
        headers = list(reader.fieldnames or [])
    return rows, headers


def pick(headers, preferred, variants):
    if preferred and preferred in headers:
        return preferred
    for name in variants:
        if name in headers:
            return name
    return ""


def to_float(value):
    raw = str(value or "").strip().replace(",", ".")
    if raw.endswith("%"):
        raw = raw[:-1]
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def to_bool(value):
    raw = str(value or "").strip().lower()
    if raw in {"1", "true", "yes", "y", "error", "wrong", "failed"}:
        return True
    if raw in {"0", "false", "no", "n", "ok", "correct", "passed"}:
        return False
    number = to_float(raw)
    if number is None:
        return None
    return number > 0


def to_percent(value):
    number = to_float(value)
    if number is None:
        return None
    if 0 <= number <= 1:
        return number * 100
    return number


def norm_label(value):
    raw = str(value or "").strip()
    if raw.lower() in {"", "none", "null", "nan", "-"}:
        return ""
    return raw


def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    values = sorted(values)
    pos = (len(values) - 1) * p
    left = math.floor(pos)
    right = math.ceil(pos)
    if left == right:
        return float(values[left])
    k = pos - left
    return float(values[left] + (values[right] - values[left]) * k)


def rounded(value):
    if value is None:
        return None
    return round(float(value), 6)


def reduction(a, b):
    if a is None or b is None or a == 0:
        return None
    return round((a - b) / a * 100.0, 6)


def simple_stats(time_values, error_values, load_values, samples):
    return {
        "samples": samples,
        "avg_time_sec": rounded(statistics.mean(time_values) if time_values else None),
        "median_time_sec": rounded(statistics.median(time_values) if time_values else None),
        "p95_time_sec": rounded(percentile(time_values, 0.95)),
        "error_rate": rounded(statistics.mean(error_values) if error_values else None),
        "avg_operator_load_pct": rounded(statistics.mean(load_values) if load_values else None),
        "time_values_count": len(time_values),
        "error_values_count": len(error_values),
        "operator_load_values_count": len(load_values),
    }


def task_metrics(y_true, y_pred):
    if not y_true:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "labels": {},
            "confusion": {},
        }

    labels = sorted(set(y_true) | set(y_pred))
    matrix = {label: Counter() for label in labels}
    for true_value, pred_value in zip(y_true, y_pred):
        matrix[true_value][pred_value] += 1

    correct = sum(1 for true_value, pred_value in zip(y_true, y_pred) if true_value == pred_value)
    precisions = []
    recalls = []
    f1s = []
    weighted_sum = 0.0
    total_support = 0
    labels_report = {}

    for label in labels:
        tp = float(matrix[label][label])
        fp = float(sum(matrix[x][label] for x in labels if x != label))
        fn = float(sum(matrix[label][x] for x in labels if x != label))
        support = int(sum(matrix[label].values()))

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        labels_report[label] = {
            "support": support,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        weighted_sum += f1 * support
        total_support += support

    confusion = {}
    for true_label, row in matrix.items():
        if sum(row.values()) > 0:
            confusion[true_label] = dict(sorted(row.items(), key=lambda item: item[1], reverse=True))

    return {
        "samples": len(y_true),
        "accuracy": round(correct / len(y_true), 6),
        "macro_precision": round(sum(precisions) / len(precisions), 6),
        "macro_recall": round(sum(recalls) / len(recalls), 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "weighted_f1": round(weighted_sum / max(1, total_support), 6),
        "labels": labels_report,
        "confusion": confusion,
    }


def build_task_report(rows, true_col, pred_col):
    if not true_col or not pred_col:
        return None
    y_true = []
    y_pred = []
    skipped = 0
    for row in rows:
        true_value = norm_label(row.get(true_col))
        if not true_value:
            skipped += 1
            continue
        pred_value = norm_label(row.get(pred_col)) or EMPTY
        y_true.append(true_value)
        y_pred.append(pred_value)
    return {
        "true_col": true_col,
        "pred_col": pred_col,
        "skipped_unlabeled": skipped,
        **task_metrics(y_true, y_pred),
    }


def print_group_report(report):
    print("\n== A/B Test ==")
    print(f"mode={report['mode']}")
    for group_name, stats in (report.get("groups") or {}).items():
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


def print_task(title, report):
    print(f"\n== {title} ==")
    print(f"columns: true='{report['true_col']}', pred='{report['pred_col']}'")
    print(f"samples={report['samples']}, skipped_unlabeled={report['skipped_unlabeled']}")
    print(
        "accuracy={acc:.4f}, macro_p={p:.4f}, macro_r={r:.4f}, macro_f1={f1:.4f}, weighted_f1={wf1:.4f}".format(
            acc=float(report["accuracy"]),
            p=float(report["macro_precision"]),
            r=float(report["macro_recall"]),
            f1=float(report["macro_f1"]),
            wf1=float(report["weighted_f1"]),
        )
    )


def build_long_report(args, rows, headers, csv_path):
    group_col = pick(headers, args.group_col, ["group", "variant", "ab_group"])
    time_col = pick(headers, args.time_col, ["classification_time_sec", "time_sec", "duration_sec"])
    error_col = pick(headers, args.error_col, ["is_error", "error", "wrong_category"])
    load_col = pick(headers, args.operator_load_col, ["operator_load_pct", "operator_involvement_pct", "operator_share_pct", "operator_load"])

    if not group_col:
        raise RuntimeError("group column not found")

    groups = {}
    for row in rows:
        name = str(row.get(group_col) or "").strip() or "unknown"
        groups.setdefault(name, {"times": [], "errors": [], "loads": [], "samples": 0})
        groups[name]["samples"] += 1

        value = to_float(row.get(time_col)) if time_col else None
        if value is not None:
            groups[name]["times"].append(value)

        value = to_bool(row.get(error_col)) if error_col else None
        if value is not None:
            groups[name]["errors"].append(1.0 if value else 0.0)

        value = to_percent(row.get(load_col)) if load_col else None
        if value is not None:
            groups[name]["loads"].append(value)

    result_groups = {}
    for name, item in sorted(groups.items()):
        result_groups[name] = simple_stats(item["times"], item["errors"], item["loads"], item["samples"])

    a = result_groups.get(args.group_a)
    b = result_groups.get(args.group_b)
    comparison = {}
    if a and b:
        comparison = {
            "control_group": args.group_a,
            "treatment_group": args.group_b,
            "time_reduction_pct": reduction(a["avg_time_sec"], b["avg_time_sec"]),
            "error_reduction_pct": reduction(a["error_rate"], b["error_rate"]),
            "operator_load_reduction_pct": reduction(a["avg_operator_load_pct"], b["avg_operator_load_pct"]),
        }

    return {
        "mode": "long",
        "csv": str(csv_path),
        "rows_total": len(rows),
        "group_col": group_col,
        "time_col": time_col,
        "error_col": error_col,
        "operator_load_col": load_col,
        "groups": result_groups,
        "comparison": comparison,
    }


def build_side_stats(rows, time_col, load_col, default_load):
    times = []
    loads = []
    for row in rows:
        value = to_float(row.get(time_col)) if time_col else None
        if value is not None:
            times.append(value)
        value = to_percent(row.get(load_col)) if load_col else default_load
        if value is not None:
            loads.append(float(value))
    return simple_stats(times, [], loads, len(rows))


def build_paired_report(args, rows, headers, csv_path):
    manual_time_col = pick(headers, args.manual_time_col, ["manual_time_sec", "operator_time_sec", "manual_duration_sec"])
    system_time_col = pick(headers, args.system_time_col, ["system_time_sec", "classification_time_sec", "ai_time_sec", "processing_routing_sec", "processing_total_sec"])
    manual_load_col = pick(headers, args.manual_load_col, ["manual_operator_load_pct", "manual_load_pct"])
    system_load_col = pick(headers, args.system_load_col, ["system_operator_load_pct", "system_load_pct"])

    manual_intent_col = pick(headers, args.manual_intent_col, ["final_intent_id", "manual_intent", "true_intent"])
    system_intent_col = pick(headers, args.system_intent_col, ["ai_intent_id", "pred_intent", "system_intent"])
    manual_group_col = pick(headers, args.manual_group_col, ["final_group_id", "manual_group"])
    system_group_col = pick(headers, args.system_group_col, ["ai_group_id", "suggested_group", "system_group"])
    manual_priority_col = pick(headers, args.manual_priority_col, ["final_priority", "manual_priority"])
    system_priority_col = pick(headers, args.system_priority_col, ["ai_priority", "priority", "system_priority"])

    ref_intent_col = pick(headers, args.reference_intent_col, [])
    ref_group_col = pick(headers, args.reference_group_col, [])
    ref_priority_col = pick(headers, args.reference_priority_col, [])

    groups = {
        args.manual_label: build_side_stats(rows, manual_time_col, manual_load_col, args.default_manual_load_pct if not manual_load_col else None),
        args.system_label: build_side_stats(rows, system_time_col, system_load_col, args.default_system_load_pct if not system_load_col else None),
    }

    agreement = {}
    for name, true_col, pred_col in [
        ("intent", manual_intent_col, system_intent_col),
        ("group", manual_group_col, system_group_col),
        ("priority", manual_priority_col, system_priority_col),
    ]:
        report = build_task_report(rows, true_col, pred_col)
        if report:
            agreement[name] = report

    reference_quality = {}
    for name, ref_col, manual_col, system_col in [
        ("intent", ref_intent_col, manual_intent_col, system_intent_col),
        ("group", ref_group_col, manual_group_col, system_group_col),
        ("priority", ref_priority_col, manual_priority_col, system_priority_col),
    ]:
        if not ref_col:
            continue
        manual_report = build_task_report(rows, ref_col, manual_col)
        system_report = build_task_report(rows, ref_col, system_col)
        if manual_report and system_report:
            reference_quality[name] = {
                args.manual_label: manual_report,
                args.system_label: system_report,
            }

    return {
        "mode": "paired",
        "csv": str(csv_path),
        "rows_total": len(rows),
        "manual_label": args.manual_label,
        "system_label": args.system_label,
        "columns": {
            "manual_time_col": manual_time_col,
            "system_time_col": system_time_col,
            "manual_load_col": manual_load_col,
            "system_load_col": system_load_col,
            "manual_intent_col": manual_intent_col,
            "system_intent_col": system_intent_col,
            "manual_group_col": manual_group_col,
            "system_group_col": system_group_col,
            "manual_priority_col": manual_priority_col,
            "system_priority_col": system_priority_col,
            "reference_intent_col": ref_intent_col,
            "reference_group_col": ref_group_col,
            "reference_priority_col": ref_priority_col,
        },
        "groups": groups,
        "comparison": {
            "control_group": args.manual_label,
            "treatment_group": args.system_label,
            "time_reduction_pct": reduction(groups[args.manual_label]["avg_time_sec"], groups[args.system_label]["avg_time_sec"]),
            "error_reduction_pct": None,
            "operator_load_reduction_pct": reduction(groups[args.manual_label]["avg_operator_load_pct"], groups[args.system_label]["avg_operator_load_pct"]),
        },
        "agreement": agreement,
        "reference_quality": reference_quality,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Simple A/B metrics from CSV.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--mode", choices=["long", "paired"], default="long")
    parser.add_argument("--out-json", default="")

    parser.add_argument("--group-col", default="group")
    parser.add_argument("--time-col", default="classification_time_sec")
    parser.add_argument("--error-col", default="is_error")
    parser.add_argument("--operator-load-col", default="operator_load_pct")
    parser.add_argument("--group-a", default="A")
    parser.add_argument("--group-b", default="B")

    parser.add_argument("--manual-label", default="manual")
    parser.add_argument("--system-label", default="system")
    parser.add_argument("--manual-time-col", default="manual_time_sec")
    parser.add_argument("--system-time-col", default="system_time_sec")
    parser.add_argument("--manual-load-col", default="manual_operator_load_pct")
    parser.add_argument("--system-load-col", default="system_operator_load_pct")
    parser.add_argument("--default-manual-load-pct", type=float, default=100.0)
    parser.add_argument("--default-system-load-pct", type=float, default=0.0)

    parser.add_argument("--manual-intent-col", default="final_intent_id")
    parser.add_argument("--system-intent-col", default="ai_intent_id")
    parser.add_argument("--manual-group-col", default="final_group_id")
    parser.add_argument("--system-group-col", default="ai_group_id")
    parser.add_argument("--manual-priority-col", default="final_priority")
    parser.add_argument("--system-priority-col", default="ai_priority")

    parser.add_argument("--reference-intent-col", default="")
    parser.add_argument("--reference-group-col", default="")
    parser.add_argument("--reference-priority-col", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    rows, headers = read_csv(csv_path)
    if not rows:
        print(f"[ERROR] csv is empty: {csv_path}")
        return 2

    try:
        if args.mode == "paired":
            report = build_paired_report(args, rows, headers, csv_path)
        else:
            report = build_long_report(args, rows, headers, csv_path)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        print("headers:", ", ".join(headers))
        return 2

    print_group_report(report)

    if args.mode == "paired":
        for name, item in (report.get("agreement") or {}).items():
            print_task(f"Agreement: {name.capitalize()}", item)
        for name, item in (report.get("reference_quality") or {}).items():
            for side_name, side_report in item.items():
                print_task(f"Reference Quality: {name.capitalize()} ({side_name})", side_report)

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (csv_path.parent / "ab_test_metrics.json")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] report saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
