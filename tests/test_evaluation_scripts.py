from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


class EvaluationScriptsTest(unittest.TestCase):
    maxDiff = None

    def _run_script(self, script_name: str, *args: str) -> subprocess.CompletedProcess[str]:
        script_path = SCRIPTS_DIR / script_name
        return subprocess.run(
            [sys.executable, str(script_path), *args],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_evaluate_routing_csv_produces_expected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "routing.csv"
            rows = [
                {
                    "final_intent_id": "billing",
                    "ai_intent_id": "billing",
                    "final_group_id": "support",
                    "ai_group_id": "support",
                    "final_priority": "high",
                    "ai_priority": "high",
                },
                {
                    "final_intent_id": "delivery",
                    "ai_intent_id": "billing",
                    "final_group_id": "support",
                    "ai_group_id": "support",
                    "final_priority": "medium",
                    "ai_priority": "low",
                },
                {
                    "final_intent_id": "billing",
                    "ai_intent_id": "",
                    "final_group_id": "finance",
                    "ai_group_id": "finance",
                    "final_priority": "low",
                    "ai_priority": "low",
                },
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            result = self._run_script("evaluate_routing_csv.py", "--csv", str(csv_path))
            self.assertIn("== Intent ==", result.stdout)

            report = json.loads((Path(tmpdir) / "routing_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["rows_total"], 3)
            self.assertAlmostEqual(report["intent"]["accuracy"], 0.333333, places=6)
            self.assertAlmostEqual(report["group"]["accuracy"], 1.0, places=6)
            self.assertAlmostEqual(report["priority"]["accuracy"], 0.666667, places=6)

    def test_evaluate_transcription_wer_reports_word_and_char_error_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "transcription.csv"
            rows = [
                {"source_file": "call_1.wav", "reference_text": "Привет мир", "hypothesis_text": "Привет мир"},
                {"source_file": "call_2.wav", "reference_text": "заказ доставлен вчера", "hypothesis_text": "заказ доставлен"},
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            result = self._run_script("evaluate_transcription_wer.py", "--csv", str(csv_path))
            self.assertIn("== Transcription Quality ==", result.stdout)

            report = json.loads((Path(tmpdir) / "transcription_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(report["samples_scored"], 2)
            self.assertAlmostEqual(report["wer"], 0.2, places=6)
            self.assertEqual(report["word_totals"]["deletions"], 1)
            self.assertEqual(report["perfect_matches"], 1)

    def test_evaluate_ab_test_compares_control_and_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "ab.csv"
            rows = [
                {"group": "A", "classification_time_sec": "20", "is_error": "1", "operator_load_pct": "100"},
                {"group": "A", "classification_time_sec": "22", "is_error": "0", "operator_load_pct": "100"},
                {"group": "B", "classification_time_sec": "4.0", "is_error": "0", "operator_load_pct": "56"},
                {"group": "B", "classification_time_sec": "4.6", "is_error": "0", "operator_load_pct": "56"},
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            result = self._run_script("evaluate_ab_test.py", "--csv", str(csv_path))
            self.assertIn("== A/B Test ==", result.stdout)

            report = json.loads((Path(tmpdir) / "ab_test_metrics.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(report["groups"]["A"]["avg_time_sec"], 21.0, places=6)
            self.assertAlmostEqual(report["groups"]["B"]["avg_time_sec"], 4.3, places=6)
            self.assertAlmostEqual(report["groups"]["A"]["error_rate"], 0.5, places=6)
            self.assertAlmostEqual(report["groups"]["B"]["avg_operator_load_pct"], 56.0, places=6)
            self.assertAlmostEqual(report["comparison"]["time_reduction_pct"], 79.52381, places=5)
            self.assertAlmostEqual(report["comparison"]["operator_load_reduction_pct"], 44.0, places=6)


if __name__ == "__main__":
    unittest.main()
