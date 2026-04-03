#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


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


def _normalize_text(value: object, case_sensitive: bool, keep_punctuation: bool) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00A0", " ")
    if not case_sensitive:
        text = text.lower()
    if not keep_punctuation:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _align_counts(reference: Sequence[str], hypothesis: Sequence[str]) -> Dict[str, int]:
    n = len(reference)
    m = len(hypothesis)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = "delete"
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = "insert"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            same = reference[i - 1] == hypothesis[j - 1]
            candidates = [
                (dp[i - 1][j] + 1, "delete"),
                (dp[i][j - 1] + 1, "insert"),
                (dp[i - 1][j - 1] + (0 if same else 1), "match" if same else "substitute"),
            ]
            best_cost, best_op = min(candidates, key=lambda item: (item[0], 0 if item[1] == "match" else 1))
            dp[i][j] = best_cost
            back[i][j] = best_op

    substitutions = 0
    deletions = 0
    insertions = 0
    i = n
    j = m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "match":
            i -= 1
            j -= 1
        elif op == "substitute":
            substitutions += 1
            i -= 1
            j -= 1
        elif op == "delete":
            deletions += 1
            i -= 1
        elif op == "insert":
            insertions += 1
            j -= 1
        else:
            break

    return {
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "distance": dp[n][m],
    }


def _score_pair(reference: str, hypothesis: str) -> Dict[str, object]:
    ref_words = reference.split() if reference else []
    hyp_words = hypothesis.split() if hypothesis else []
    word_counts = _align_counts(ref_words, hyp_words)
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    char_counts = _align_counts(ref_chars, hyp_chars)

    ref_word_count = len(ref_words)
    ref_char_count = len(ref_chars)

    return {
        "reference_words": ref_word_count,
        "hypothesis_words": len(hyp_words),
        "reference_chars": ref_char_count,
        "hypothesis_chars": len(hyp_chars),
        "wer": round(word_counts["distance"] / ref_word_count, 6) if ref_word_count > 0 else 0.0,
        "cer": round(char_counts["distance"] / ref_char_count, 6) if ref_char_count > 0 else 0.0,
        "word_counts": word_counts,
        "char_counts": char_counts,
    }


def _print_summary(report: Dict[str, object], ref_col: str, hyp_col: str) -> None:
    print("\n== Transcription Quality ==")
    print(f"columns: ref='{ref_col}', hyp='{hyp_col}'")
    print(
        "samples={samples}, skipped_empty_reference={skipped}, perfect_matches={perfect}".format(
            samples=report["samples_scored"],
            skipped=report["skipped_empty_reference"],
            perfect=int(report["perfect_matches"]),
        )
    )
    print(
        "WER={wer:.4f}, CER={cer:.4f}, avg_row_WER={avg_row_wer:.4f}, perfect_match_rate={pmr:.4f}".format(
            wer=float(report["wer"]),
            cer=float(report["cer"]),
            avg_row_wer=float(report["avg_row_wer"]),
            pmr=float(report["perfect_match_rate"]),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR quality from CSV using WER/CER.")
    parser.add_argument("--csv", required=True, help="Path to CSV with reference and hypothesis texts.")
    parser.add_argument("--ref-col", default="reference_text", help="Column with reference transcript.")
    parser.add_argument("--hyp-col", default="hypothesis_text", help="Column with ASR transcript.")
    parser.add_argument("--id-col", default="source_file", help="Optional identifier column used in the report.")
    parser.add_argument("--out-json", default="", help="Optional path to save JSON report.")
    parser.add_argument("--case-sensitive", action="store_true", help="Disable lowercase normalization.")
    parser.add_argument("--keep-punctuation", action="store_true", help="Keep punctuation during normalization.")
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

    ref_col = _pick_column(headers, args.ref_col, ["reference_text", "reference", "ground_truth_text", "final_transcript"])
    hyp_col = _pick_column(headers, args.hyp_col, ["hypothesis_text", "predicted_text", "transcript_text", "asr_text"])
    id_col = args.id_col if args.id_col in headers else ""
    if not ref_col or not hyp_col:
        print("[ERROR] missing required columns")
        print("headers:", ", ".join(headers))
        return 2

    total_ref_words = 0
    total_ref_chars = 0
    total_word_distance = 0
    total_char_distance = 0
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_char_substitutions = 0
    total_char_deletions = 0
    total_char_insertions = 0
    skipped = 0
    perfect_matches = 0
    row_wers: List[float] = []
    per_sample: List[Dict[str, object]] = []

    for index, row in enumerate(rows, start=1):
        ref_text = _normalize_text(row.get(ref_col, ""), args.case_sensitive, args.keep_punctuation)
        if not ref_text:
            skipped += 1
            continue
        hyp_text = _normalize_text(row.get(hyp_col, ""), args.case_sensitive, args.keep_punctuation)
        score = _score_pair(ref_text, hyp_text)

        word_counts = score["word_counts"]
        char_counts = score["char_counts"]
        total_ref_words += int(score["reference_words"])
        total_ref_chars += int(score["reference_chars"])
        total_word_distance += int(word_counts["distance"])
        total_char_distance += int(char_counts["distance"])
        total_substitutions += int(word_counts["substitutions"])
        total_deletions += int(word_counts["deletions"])
        total_insertions += int(word_counts["insertions"])
        total_char_substitutions += int(char_counts["substitutions"])
        total_char_deletions += int(char_counts["deletions"])
        total_char_insertions += int(char_counts["insertions"])
        wer = float(score["wer"])
        row_wers.append(wer)
        if word_counts["distance"] == 0:
            perfect_matches += 1

        sample_id = str(row.get(id_col, "")).strip() if id_col else ""
        per_sample.append(
            {
                "row_index": index,
                "sample_id": sample_id,
                "wer": round(wer, 6),
                "cer": float(score["cer"]),
                "reference_words": int(score["reference_words"]),
                "hypothesis_words": int(score["hypothesis_words"]),
            }
        )

    if total_ref_words == 0:
        print("[ERROR] no rows with non-empty reference transcript")
        return 2

    per_sample_sorted = sorted(
        per_sample,
        key=lambda item: (-float(item["wer"]), -float(item["cer"]), -int(item["reference_words"]), int(item["row_index"])),
    )

    report = {
        "csv": str(csv_path),
        "samples_total": len(rows),
        "samples_scored": len(row_wers),
        "skipped_empty_reference": skipped,
        "wer": round(total_word_distance / total_ref_words, 6),
        "cer": round(total_char_distance / total_ref_chars, 6) if total_ref_chars > 0 else 0.0,
        "avg_row_wer": round(sum(row_wers) / len(row_wers), 6),
        "perfect_matches": perfect_matches,
        "perfect_match_rate": round(perfect_matches / len(row_wers), 6),
        "word_totals": {
            "reference": total_ref_words,
            "distance": total_word_distance,
            "substitutions": total_substitutions,
            "deletions": total_deletions,
            "insertions": total_insertions,
        },
        "char_totals": {
            "reference": total_ref_chars,
            "distance": total_char_distance,
            "substitutions": total_char_substitutions,
            "deletions": total_char_deletions,
            "insertions": total_char_insertions,
        },
        "worst_samples": per_sample_sorted[:20],
    }

    _print_summary(report, ref_col, hyp_col)

    out_json = (
        Path(args.out_json).expanduser().resolve()
        if args.out_json
        else (csv_path.parent / "transcription_metrics.json")
    )
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] report saved: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
