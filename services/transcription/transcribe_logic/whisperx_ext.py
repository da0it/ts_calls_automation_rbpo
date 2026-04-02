from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from transcribe_logic.whisperx_device import resolve_whisperx_device


def _run(cmd: List[str], timeout_sec: int) -> None:
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"WhisperX backend timed out after {timeout_sec}s: {' '.join(cmd)}"
        ) from exc
    if p.returncode != 0:
        raise RuntimeError(
            "WhisperX backend failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n" + p.stdout
            + "\n\nSTDERR:\n" + p.stderr
        )


def whisperx_transcribe_via_cli(
    audio_path: str,
    *,
    venv_python: str,
    model: str = "large-v3",
    language: str = "ru",
    device: str = "auto",
    compute_type: str = "int8",
    batch_size: int = 1,
    vad_method: str = "silero",
) -> List[Dict[str, Any]]:
    device = resolve_whisperx_device(device)
    worker = Path(__file__).with_name("whisperx_worker.py")
    if not os.path.exists(venv_python):
        raise RuntimeError(f"whisperx venv python not found: {venv_python}")
    if not worker.exists():
        raise RuntimeError(f"whisperx worker script not found: {worker}")

    timeout_sec = int(os.getenv("WHISPERX_TIMEOUT_SECONDS", "2400"))
    with tempfile.TemporaryDirectory() as td:
        out_json = os.path.join(td, "whisperx_segments.json")
        cmd = [
            venv_python,
            str(worker),
            "--audio",
            audio_path,
            "--out-json",
            out_json,
            "--model",
            model,
            "--language",
            language,
            "--device",
            device,
            "--compute-type",
            compute_type,
            "--batch-size",
            str(batch_size),
            "--vad-method",
            vad_method,
        ]

        _run(cmd, timeout_sec=timeout_sec)

        with open(out_json, "r", encoding="utf-8") as f:
            payload = json.load(f)

    segments = payload.get("segments", [])
    if not isinstance(segments, list):
        raise RuntimeError("WhisperX output has invalid format: 'segments' is not a list")
    return segments
