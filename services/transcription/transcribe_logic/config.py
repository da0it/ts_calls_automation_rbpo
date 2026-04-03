# transcribe/config.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict

AUTO_DEVICE = "auto"

@dataclass
class AudioCfg:
    sample_rate: int = 16000
    mono_channels: int = 1

    highpass_hz: int = 80
    lowpass_hz: int = 7900

    cut_codec: str = "pcm_s16le"

@dataclass
class CutCfg:
    pad_seconds: float = 0.0

@dataclass
class SilenceCfg:
    # silencedetect params
    silence_db: float = -35.0
    silence_min_dur: float = 0.25

    # splitting by silence
    split_max_len: float = 4.0
    split_pad: float = 0.05
    edge_guard_seconds: float = 0.3
    min_piece_seconds: float = 0.4

@dataclass
class TurnsCfg:
    merge_max_gap: float = 0.1
    merge_min_dur: float = 0.25

    long_turn_max_len: float = 6
    long_turn_overlap: float = 0.2

    # merge already-transcribed utterances in final timeline
    merge_utt_max_gap: float = 0.7

@dataclass
class ASRCfg:
    # minimal duration to process
    min_dur: float = 0.25

    # ASR inner splitting by silences
    silence_db: float = -35.0
    silence_min_dur: float = 0.15
    piece_max_len: float = 4
    piece_pad: float = 0.05

@dataclass
class StereoCfg:
    threshold: float = 0.98
    rms_diff_db: float = 1.0

@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    cut: CutCfg = field(default_factory=CutCfg)
    silence: SilenceCfg = field(default_factory=SilenceCfg)
    turns: TurnsCfg = field(default_factory=TurnsCfg)
    asr: ASRCfg = field(default_factory=ASRCfg)
    stereo: StereoCfg = field(default_factory=StereoCfg)


def normalize_whisperx_device(device: str | None) -> str:
    value = (device or "").strip().lower()
    if not value:
        return AUTO_DEVICE
    if value == "gpu":
        return "cuda"
    return value


@lru_cache(maxsize=1)
def _cuda_is_available() -> bool:
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_whisperx_device(device: str | None = None) -> str:
    device = normalize_whisperx_device(device)
    if device != AUTO_DEVICE:
        return device
    return "cuda" if _cuda_is_available() else "cpu"


def get_whisperx_device_from_env(name: str = "WHISPERX_DEVICE") -> str:
    return resolve_whisperx_device(os.getenv(name, AUTO_DEVICE))


def get_whisperx_settings() -> Dict[str, Any]:
    vad_method = os.getenv("WHISPERX_VAD_METHOD", "silero").strip().lower()
    if not vad_method:
        vad_method = "silero"

    return {
        "model": os.getenv("WHISPERX_MODEL", "large-v3"),
        "language": os.getenv("WHISPERX_LANGUAGE", "ru"),
        "device": get_whisperx_device_from_env(),
        "compute_type": os.getenv("WHISPERX_COMPUTE_TYPE", "int8"),
        "batch_size": int(os.getenv("WHISPERX_BATCH_SIZE", "1")),
        "vad_method": vad_method,
    }


CFG = Config()
