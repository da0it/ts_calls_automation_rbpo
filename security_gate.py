#!/usr/bin/env python3
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


SEVERITY_ORDER = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def bandit_issues(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    issues = []
    for item in data.get("results", []):
        issues.append(
            {
                "tool": "bandit",
                "severity": str(item.get("issue_severity", "LOW")).upper(),
                "rule": item.get("test_id", "?"),
                "file": item.get("filename", "<unknown>"),
                "line": item.get("line_number", "?"),
                "details": item.get("issue_text", ""),
            }
        )
    return issues


def gosec_issues(path: Path) -> list[dict[str, Any]]:
    data = read_json(path)
    issues = []
    for item in data.get("Issues", []):
        issues.append(
            {
                "tool": "gosec",
                "severity": str(item.get("severity", "LOW")).upper(),
                "rule": item.get("rule_id", "?"),
                "file": item.get("file", "<unknown>"),
                "line": item.get("line", "?"),
                "details": item.get("details", ""),
            }
        )
    return issues


def is_blocking(issue: dict[str, Any], threshold: str) -> bool:
    return SEVERITY_ORDER.get(issue["severity"], 0) >= SEVERITY_ORDER[threshold]


def print_summary(name: str, issues: list[dict[str, Any]]) -> None:
    severities = Counter(issue["severity"] for issue in issues)
    print(
        f"{name}: total={len(issues)} "
        f"LOW={severities.get('LOW', 0)} "
        f"MEDIUM={severities.get('MEDIUM', 0)} "
        f"HIGH={severities.get('HIGH', 0)} "
        f"CRITICAL={severities.get('CRITICAL', 0)}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Security Gate for Bandit and gosec reports")
    parser.add_argument("--bandit", action="append", default=[], type=Path, help="Path to Bandit JSON report")
    parser.add_argument("--gosec", action="append", default=[], type=Path, help="Path to gosec JSON report")
    parser.add_argument(
        "--threshold",
        choices=("low", "medium", "high", "critical"),
        default="high",
        help="Minimal severity that blocks the pipeline",
    )
    args = parser.parse_args()

    threshold = args.threshold.upper()
    all_issues: list[dict[str, Any]] = []

    for path in args.bandit:
        issues = bandit_issues(path)
        print_summary(path.name, issues)
        all_issues.extend(issues)

    for path in args.gosec:
        issues = gosec_issues(path)
        print_summary(path.name, issues)
        all_issues.extend(issues)

    blockers = [issue for issue in all_issues if is_blocking(issue, threshold)]
    print(f"Security Gate threshold: {threshold}")
    print(f"All issues: {len(all_issues)}")
    print(f"Blocking issues: {len(blockers)}")

    for issue in blockers:
        print(
            f"{issue['tool']} {issue['severity']} {issue['rule']} "
            f"{issue['file']}:{issue['line']} {issue['details']}"
        )

    return 1 if blockers else 0


if __name__ == "__main__":
    sys.exit(main())
