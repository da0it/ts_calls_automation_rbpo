#!/usr/bin/env python3
"""Build benchmark charts for local LLM model comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


CASE_ORDER = ["small", "medium", "large"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot charts from benchmark_ollama_summary.py JSON output."
    )
    parser.add_argument(
        "--input-json",
        dest="input_json",
        action="append",
        required=True,
        help="Path to benchmark results JSON. May be passed multiple times.",
    )
    parser.add_argument(
        "--outdir",
        default="benchmark_plots",
        help="Directory where charts will be saved.",
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional title prefix for all charts.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help="Optional model filter. May be passed multiple times.",
    )
    return parser.parse_args()


def load_results(paths: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON array in {path}")
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError(f"Expected result object in {path}")
            item["_source_file"] = str(path)
            rows.append(item)
    return rows


def filter_results(rows: list[dict[str, Any]], models: set[str] | None) -> list[dict[str, Any]]:
    if not models:
        return rows
    return [row for row in rows if str(row.get("model")) in models]


def safe_mean(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return statistics.mean(clean)


def cyrillic_ratio(text: str) -> float:
    cyr = 0
    lat = 0
    for char in text:
        lower = char.lower()
        if "а" <= lower <= "я" or lower == "ё":
            cyr += 1
        elif "a" <= lower <= "z":
            lat += 1
    total = cyr + lat
    if total == 0:
        return 0.0
    return cyr / total


def is_russian_localized(text: str) -> bool:
    if not text:
        return False
    return cyrillic_ratio(text) >= 0.5


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    metrics: list[dict[str, Any]] = []
    for model, items in sorted(grouped.items()):
        json_ok = sum(1 for item in items if bool(item.get("valid_json")))
        problem_ok = sum(1 for item in items if bool(item.get("non_empty_problem")))
        localized_ok = sum(
            1
            for item in items
            if bool(item.get("non_empty_problem")) and is_russian_localized(str(item.get("problem") or ""))
        )
        large_items = [item for item in items if str(item.get("case_size")) == "large"]

        metrics.append(
            {
                "model": model,
                "runs": len(items),
                "wall_avg_s": safe_mean(
                    [float(item["wall_clock_sec"]) for item in items if item.get("wall_clock_sec") is not None]
                ),
                "tok_s_avg": safe_mean(
                    [float(item["eval_tokens_per_sec"]) for item in items if item.get("eval_tokens_per_sec") is not None]
                ),
                "completeness_avg": safe_mean(
                    [float(item["completeness_score"]) for item in items if item.get("completeness_score") is not None]
                ),
                "json_ok_rate": json_ok / len(items) if items else 0.0,
                "problem_ok_rate": problem_ok / len(items) if items else 0.0,
                "localized_rate": localized_ok / len(items) if items else 0.0,
                "large_completeness": safe_mean(
                    [
                        float(item["completeness_score"])
                        for item in large_items
                        if item.get("completeness_score") is not None
                    ]
                )
                or 0.0,
            }
        )
    return metrics


def aggregate_by_model_and_size(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | None]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["model"])][str(row["case_size"])].append(row)

    result: dict[str, dict[str, dict[str, float | None]]] = {}
    for model, size_map in grouped.items():
        result[model] = {}
        for size in CASE_ORDER:
            items = size_map.get(size, [])
            result[model][size] = {
                "wall_avg_s": safe_mean(
                    [float(item["wall_clock_sec"]) for item in items if item.get("wall_clock_sec") is not None]
                ),
                "completeness_avg": safe_mean(
                    [
                        float(item["completeness_score"])
                        for item in items
                        if item.get("completeness_score") is not None
                    ]
                ),
            }
    return result


def normalize_inverse(values: list[float]) -> list[float]:
    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if math.isclose(min_value, max_value):
        return [1.0 for _ in values]
    return [(max_value - value) / (max_value - min_value) for value in values]


def title(prefix: str, text: str) -> str:
    return f"{prefix} {text}".strip()


def plot_speed_vs_quality(
    metrics: list[dict[str, Any]],
    outdir: Path,
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    xs = [float(item["wall_avg_s"]) for item in metrics if item.get("wall_avg_s") is not None]
    ys = [float(item["completeness_avg"]) for item in metrics if item.get("completeness_avg") is not None]
    cs = [float(item["localized_rate"]) for item in metrics]
    sizes = [250 + 250 * float(item["problem_ok_rate"]) for item in metrics]

    scatter = ax.scatter(xs, ys, c=cs, s=sizes, cmap="viridis", edgecolors="black", alpha=0.85)
    for item in metrics:
        ax.annotate(
            item["model"],
            (float(item["wall_avg_s"]), float(item["completeness_avg"])),
            textcoords="offset points",
            xytext=(8, 6),
            fontsize=9,
        )

    ax.set_title(title(prefix, "Скорость против качества суммаризации"))
    ax.set_xlabel("Среднее время ответа, с")
    ax.set_ylabel("Средняя полнота ответа")
    ax.grid(True, linestyle="--", alpha=0.35)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Доля русскоязычных ответов")

    fig.tight_layout()
    fig.savefig(outdir / "01_speed_vs_quality.png", dpi=220)
    plt.close(fig)


def plot_reliability_bars(
    metrics: list[dict[str, Any]],
    outdir: Path,
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    models = [item["model"] for item in metrics]
    json_rates = [100 * float(item["json_ok_rate"]) for item in metrics]
    problem_rates = [100 * float(item["problem_ok_rate"]) for item in metrics]
    localized_rates = [100 * float(item["localized_rate"]) for item in metrics]

    x = list(range(len(models)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar([value - width for value in x], json_rates, width=width, label="JSON OK, %")
    ax.bar(x, problem_rates, width=width, label="Problem OK, %")
    ax.bar([value + width for value in x], localized_rates, width=width, label="Русская локализация, %")

    ax.set_title(title(prefix, "Надёжность и соответствие требованиям"))
    ax.set_xlabel("Модель")
    ax.set_ylabel("Доля корректных ответов, %")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outdir / "02_reliability.png", dpi=220)
    plt.close(fig)


def plot_latency_by_size(
    metrics_by_size: dict[str, dict[str, dict[str, float | None]]],
    outdir: Path,
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    models = list(metrics_by_size.keys())
    x = list(range(len(models)))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, size in enumerate(CASE_ORDER):
        values = [
            metrics_by_size[model][size]["wall_avg_s"] if metrics_by_size[model][size]["wall_avg_s"] is not None else 0.0
            for model in models
        ]
        positions = [value + (idx - 1) * width for value in x]
        ax.bar(positions, values, width=width, label=size)

    ax.set_title(title(prefix, "Время ответа на кейсах разного размера"))
    ax.set_xlabel("Модель")
    ax.set_ylabel("Среднее время ответа, с")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(title="Размер кейса")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(outdir / "03_latency_by_size.png", dpi=220)
    plt.close(fig)


def plot_selection_heatmap(
    metrics: list[dict[str, Any]],
    outdir: Path,
    prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    speed_scores = normalize_inverse([float(item["wall_avg_s"]) for item in metrics])

    matrix = []
    row_labels = []
    for item, speed_score in zip(metrics, speed_scores):
        row_labels.append(item["model"])
        matrix.append(
            [
                float(item["localized_rate"]),
                float(item["json_ok_rate"]),
                float(item["problem_ok_rate"]),
                float(item["large_completeness"]),
                speed_score,
            ]
        )

    col_labels = [
        "Локализация",
        "JSON",
        "Problem",
        "Устойчивость\nна large",
        "Скорость",
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix, cmap="YlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title(prefix, "Матрица критериев выбора модели"))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Нормированная оценка")

    fig.tight_layout()
    fig.savefig(outdir / "04_selection_heatmap.png", dpi=220)
    plt.close(fig)


def save_summary_table(metrics: list[dict[str, Any]], outdir: Path) -> None:
    lines = [
        "model,wall_avg_s,tok_s_avg,completeness_avg,json_ok_rate,problem_ok_rate,localized_rate,large_completeness"
    ]
    for item in metrics:
        lines.append(
            ",".join(
                [
                    item["model"],
                    f"{float(item['wall_avg_s']):.3f}",
                    f"{float(item['tok_s_avg']):.2f}",
                    f"{float(item['completeness_avg']):.3f}",
                    f"{float(item['json_ok_rate']):.3f}",
                    f"{float(item['problem_ok_rate']):.3f}",
                    f"{float(item['localized_rate']):.3f}",
                    f"{float(item['large_completeness']):.3f}",
                ]
            )
        )
    (outdir / "summary_metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rows = load_results(args.input_json)
    rows = filter_results(rows, set(args.models) if args.models else None)
    if not rows:
        raise SystemExit("No benchmark rows selected.")

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    metrics = aggregate_by_model(rows)
    metrics_by_size = aggregate_by_model_and_size(rows)

    plot_speed_vs_quality(metrics, outdir, args.title_prefix)
    plot_reliability_bars(metrics, outdir, args.title_prefix)
    plot_latency_by_size(metrics_by_size, outdir, args.title_prefix)
    plot_selection_heatmap(metrics, outdir, args.title_prefix)
    save_summary_table(metrics, outdir)

    print(f"Saved charts to: {outdir}")
    print(f"- {outdir / '01_speed_vs_quality.png'}")
    print(f"- {outdir / '02_reliability.png'}")
    print(f"- {outdir / '03_latency_by_size.png'}")
    print(f"- {outdir / '04_selection_heatmap.png'}")
    print(f"- {outdir / 'summary_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
