from __future__ import annotations

import os
import re
import tempfile
from typing import Any, Dict, List, Optional

from transcribe_logic.audio_utils import to_wav_16k_mono_preprocessed
from transcribe_logic.whisperx_device import get_whisperx_device_from_env
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
            return f"Speaker {(numeric % 2) + 1}"
        return f"Speaker {1 if numeric <= 1 else 2 if numeric == 2 else ((numeric - 1) % 2) + 1}"

    return speaker


def _attach_basic_diarization(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    speaker_aliases: Dict[str, str] = {}
    next_speaker_index = 0

    for index, segment in enumerate(segments):
        normalized_speaker = _normalize_speaker_label(segment.get("speaker"))
        if normalized_speaker in {"Speaker 1", "Speaker 2"}:
            segment["speaker"] = normalized_speaker
        elif normalized_speaker:
            if normalized_speaker not in speaker_aliases:
                speaker_aliases[normalized_speaker] = f"Speaker {(next_speaker_index % 2) + 1}"
                next_speaker_index += 1
            segment["speaker"] = speaker_aliases[normalized_speaker]
        else:
            segment["speaker"] = f"Speaker {(index % 2) + 1}"
        segment["role"] = ""
    return segments


def transcribe_with_roles(
    audio_path: str,
    *,
    hf_token: Optional[str] = None,
    no_stem: bool = False,  # kept for backward compatibility; currently unused.
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

        common_kwargs = dict(
            model=os.getenv("WHISPERX_MODEL", "large-v3"),
            language=os.getenv("WHISPERX_LANGUAGE", "ru"),
            device=get_whisperx_device_from_env(),
            compute_type=os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
            batch_size=int(os.getenv("WHISPERX_BATCH_SIZE", "4")),
            vad_method=os.getenv("WHISPERX_VAD_METHOD", "silero").strip().lower(),
        )

        persistent = os.getenv("WHISPERX_PERSISTENT", "1").strip().lower() in {"1", "true", "yes", "on"}
        if persistent:
            segments = whisperx_transcribe_inprocess(wav, **common_kwargs)
            mode = "whisperx_persistent"
        else:
            venv_python = whisper_venv_python or _default_whisperx_venv_python()
            segments = whisperx_transcribe_via_cli(
                wav,
                venv_python=venv_python,
                **common_kwargs,
            )
            mode = "whisperx_cli"

        note = f"ASR backend whisperx ({mode}): mono 16k -> whisperx transcribe+align. Basic speaker labeling enabled."

        if not segments:
            return {
                "mode": mode,
                "input": os.path.basename(audio_path),
                "segments": [],
                "role_mapping": {},
                "note": "Backend returned no segments.",
            }

        segments = _attach_basic_diarization(segments)
        segments = _round_segments(segments, ndigits=2)
        note += " Segments are labeled with basic alternating speaker tags (Speaker 1 / Speaker 2)."

        return {
            "mode": mode,
            "input": os.path.basename(audio_path),
            "segments": segments,
            "role_mapping": {},
            "note": note,
        }
