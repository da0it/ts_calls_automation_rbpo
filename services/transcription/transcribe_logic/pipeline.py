from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from transcribe_logic.audio_utils import to_wav_16k_mono_preprocessed
from transcribe_logic.config import get_whisperx_settings
from transcribe_logic.whisperx_ext import whisperx_transcribe_via_cli
from transcribe_logic.whisperx_runtime import whisperx_transcribe_inprocess

def _default_whisperx_venv_python() -> str:
    return os.getenv(
        "WHISPERX_VENV_PYTHON",
        os.path.expanduser("~/whisperx_venv/bin/python"),
    )


def _round_segments(segments: List[Dict[str, Any]], ndigits: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in segments:
        ss = s.copy()
        if "start" in ss:
            ss["start"] = round(float(ss["start"]), ndigits)
        if "end" in ss:
            ss["end"] = round(float(ss["end"]), ndigits)
        if "text" in ss and ss["text"] is not None:
            ss["text"] = str(ss["text"]).strip()
        out.append(ss)
    out.sort(key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))
    return out


def _normalize_speaker_label(raw_speaker: Any) -> Optional[str]:
    speaker = str(raw_speaker or "").strip()
    if not speaker or speaker.upper() == "UNKNOWN":
        return None

    match = re.match(r"^speaker[\s_-]*(\d+)$", speaker, flags=re.IGNORECASE)
    if match:
        numeric = int(match.group(1))
        if "_" in speaker or speaker.upper().startswith("SPEAKER_"):
            return f"Speaker {numeric + 1}"
        return f"Speaker {numeric}"

    return speaker


def _attach_basic_diarization(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for segment in segments:
        normalized_speaker = _normalize_speaker_label(segment.get("speaker"))
        segment["speaker"] = normalized_speaker or ""
        segment.pop("role", None)
    return segments


def transcribe_with_roles(
    audio_path: str,
    *,
    hf_token: Optional[str] = None,
    no_stem: bool = False,
    whisper_repo_dir: str = "",
    whisper_venv_python: str = "",
) -> Dict[str, Any]:
    del no_stem
    del whisper_repo_dir

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    with tempfile.TemporaryDirectory() as td:
        wav = os.path.join(td, "audio_mono.wav")
        to_wav_16k_mono_preprocessed(audio_path, wav)

        common_kwargs = get_whisperx_settings()

        persistent = os.getenv("WHISPERX_PERSISTENT", "1").strip().lower() in {"1", "true", "yes", "on"}
        if persistent:
            segments = whisperx_transcribe_inprocess(wav, **common_kwargs)
            mode = "whisperx_persistent"
        else:
            venv_python = whisper_venv_python or _default_whisperx_venv_python()
            segments = whisperx_transcribe_via_cli(
                wav,
                venv_python=venv_python,
            )
            mode = "whisperx_cli"

        note = f"ASR backend whisperx ({mode}): mono 16k -> whisperx transcribe+align."

        if not segments:
            return {
                "mode": mode,
                "input": os.path.basename(audio_path),
                "segments": [],
                "note": "Backend returned no segments.",
            }

        segments = _attach_basic_diarization(segments)
        segments = _round_segments(segments, ndigits=2)
        has_speaker_labels = any(str(segment.get("speaker") or "").strip() for segment in segments)
        if has_speaker_labels:
            note += " Speaker labels are shown only when diarization data is available."
        else:
            note += " No speaker labels were produced by the backend."

        return {
            "mode": mode,
            "input": os.path.basename(audio_path),
            "segments": segments,
            "note": note,
        }
