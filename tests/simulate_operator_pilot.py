#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from math import comb, sqrt
from pathlib import Path
from statistics import mean, stdev


T_CRITICAL_DF11_ALPHA_005_TWO_TAILED = 2.201


SYNTHETIC_CASES = [
    {"case_id": 1, "scenario": "spam", "manual_time_sec": 168, "system_time_sec": 83, "manual_correct": 1, "system_correct": 1},
    {"case_id": 2, "scenario": "portal_access", "manual_time_sec": 140, "system_time_sec": 77, "manual_correct": 1, "system_correct": 1},
    {"case_id": 3, "scenario": "consulting", "manual_time_sec": 192, "system_time_sec": 88, "manual_correct": 0, "system_correct": 1},
    {"case_id": 4, "scenario": "license", "manual_time_sec": 146, "system_time_sec": 74, "manual_correct": 1, "system_correct": 1},
    {"case_id": 5, "scenario": "cooperation", "manual_time_sec": 133, "system_time_sec": 79, "manual_correct": 0, "system_correct": 1},
    {"case_id": 6, "scenario": "spam", "manual_time_sec": 181, "system_time_sec": 85, "manual_correct": 1, "system_correct": 1},
    {"case_id": 7, "scenario": "courses", "manual_time_sec": 130, "system_time_sec": 72, "manual_correct": 1, "system_correct": 1},
    {"case_id": 8, "scenario": "cooperation", "manual_time_sec": 168, "system_time_sec": 80, "manual_correct": 0, "system_correct": 0},
    {"case_id": 9, "scenario": "portal_access", "manual_time_sec": 172, "system_time_sec": 95, "manual_correct": 1, "system_correct": 1},
    {"case_id": 10, "scenario": "consulting", "manual_time_sec": 145, "system_time_sec": 76, "manual_correct": 0, "system_correct": 1},
    {"case_id": 11, "scenario": "license", "manual_time_sec": 174, "system_time_sec": 84, "manual_correct": 0, "system_correct": 1},
    {"case_id": 12, "scenario": "courses", "manual_time_sec": 143, "system_time_sec": 81, "manual_correct": 1, "system_correct": 1},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic 12-case operator effectiveness pilot.")
    parser.add_argument("--out-csv", default="operator_pilot_demo.csv")
    parser.add_argument("--out-json", default="operator_pilot_demo_summary.json")
    return parser.parse_args()


def mcnemar_exact_p_value(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def build_summary(rows: list[dict[str, int | str]]) -> dict[str, object]:
    manual_times = [int(row["manual_time_sec"]) for row in rows]
    system_times = [int(row["system_time_sec"]) for row in rows]
    differences = [m - s for m, s in zip(manual_times, system_times)]

    manual_mean = mean(manual_times)
    system_mean = mean(system_times)
    diff_mean = mean(differences)
    diff_sd = stdev(differences)
    t_stat = diff_mean / (diff_sd / sqrt(len(differences)))
    time_reduction_pct = (manual_mean - system_mean) / manual_mean * 100.0

    a = b = c = d = 0
    for row in rows:
        manual_ok = int(row["manual_correct"])
        system_ok = int(row["system_correct"])
        if manual_ok == 1 and system_ok == 1:
            a += 1
        elif manual_ok == 1 and system_ok == 0:
            b += 1
        elif manual_ok == 0 and system_ok == 1:
            c += 1
        else:
            d += 1

    manual_accuracy = sum(int(row["manual_correct"]) for row in rows) / len(rows)
    system_accuracy = sum(int(row["system_correct"]) for row in rows) / len(rows)
    mcnemar_exact_p = mcnemar_exact_p_value(b, c)
    mcnemar_chi2_cc = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) > 0 else 0.0

    return {
        "sample_size": len(rows),
        "manual_mean_time_sec": round(manual_mean, 3),
        "system_mean_time_sec": round(system_mean, 3),
        "mean_time_reduction_sec": round(diff_mean, 3),
        "time_reduction_pct": round(time_reduction_pct, 3),
        "paired_t_test": {
            "t_statistic": round(t_stat, 6),
            "df": len(rows) - 1,
            "t_critical_alpha_0_05_two_tailed": T_CRITICAL_DF11_ALPHA_005_TWO_TAILED,
            "significant_at_0_05": bool(abs(t_stat) > T_CRITICAL_DF11_ALPHA_005_TWO_TAILED),
        },
        "manual_accuracy": round(manual_accuracy, 6),
        "system_accuracy": round(system_accuracy, 6),
        "mcnemar": {
            "table": {
                "both_correct": a,
                "manual_only_correct": b,
                "system_only_correct": c,
                "both_wrong": d,
            },
            "chi_square_cc": round(mcnemar_chi2_cc, 6),
            "exact_p_value": round(mcnemar_exact_p, 6),
            "significant_at_0_05": bool(mcnemar_exact_p < 0.05),
        },
    }


def main() -> int:
    args = parse_args()
    out_csv = Path(args.out_csv).expanduser().resolve()
    out_json = Path(args.out_json).expanduser().resolve()

    rows = []
    for row in SYNTHETIC_CASES:
        item = dict(row)
        item["time_saved_sec"] = int(item["manual_time_sec"]) - int(item["system_time_sec"])
        rows.append(item)

    summary = build_summary(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "scenario",
                "manual_time_sec",
                "system_time_sec",
                "time_saved_sec",
                "manual_correct",
                "system_correct",
            ],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)

    out_json.write_text(
        json.dumps(
            {
                "kind": "synthetic_demo",
                "note": "This dataset is illustrative and should be replaced with real operator measurements.",
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[SYNTHETIC PILOT]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[OK] CSV: {out_csv}")
    print(f"[OK] JSON: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
