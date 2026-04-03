#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


EMPTY = "__empty__"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [dict(row) for row in reader]
    return rows


def clean(value):
    raw = str(value or "").strip()
    if raw.lower() in {"", "none", "null", "nan", "-"}:
        return ""
    return raw


def build_report(rows, true_col, pred_col):
    y_true = []
    y_pred = []
    skipped = 0

    for row in rows:
        true_value = clean(row.get(true_col))
        if not true_value:
            skipped += 1
            continue
        pred_value = clean(row.get(pred_col)) or EMPTY
        y_true.append(true_value)
        y_pred.append(pred_value)

    if not y_true:
        return {
            "true_col": true_col,
            "pred_col": pred_col,
            "skipped_unlabeled": skipped,
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
        "true_col": true_col,
        "pred_col": pred_col,
        "skipped_unlabeled": skipped,
        "samples": len(y_true),
        "accuracy": round(correct / len(y_true), 6),
        "macro_precision": round(sum(precisions) / len(precisions), 6),
        "macro_recall": round(sum(recalls) / len(recalls), 6),
        "macro_f1": round(sum(f1s) / len(f1s), 6),
        "weighted_f1": round(weighted_sum / max(1, total_support), 6),
        "labels": labels_report,
        "confusion": confusion,
    }


def print_report(title, report):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate routing quality from one fixed CSV format.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out-json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()

    if not csv_path.exists():
        print(f"[ERROR] csv not found: {csv_path}")
        return 2

    rows = read_csv(csv_path)
    if not rows:
        print(f"[ERROR] csv is empty: {csv_path}")
        return 2

    required = [
        "final_intent_id",
        "ai_intent_id",
        "final_group_id",
        "ai_group_id",
        "final_priority",
        "ai_priority",
    ]
    missing = [name for name in required if name not in rows[0]]
    if missing:
        print("[ERROR] missing columns:", ", ".join(missing))
        return 2

    intent_report = build_report(rows, "final_intent_id", "ai_intent_id")
    group_report = build_report(rows, "final_group_id", "ai_group_id")
    priority_report = build_report(rows, "final_priority", "ai_priority")

    print_report("Intent", intent_report)
    print_report("Group", group_report)
    print_report("Priority", priority_report)

    report = {
        "csv": str(csv_path),
        "rows_total": len(rows),
        "intent": intent_report,
        "group": group_report,
        "priority": priority_report,
    }

    out_json = Path(args.out_json).expanduser().resolve() if args.out_json else (csv_path.parent / "routing_metrics.json")
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] report saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
