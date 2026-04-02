from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

from transcribe_logic.whisperx_device import resolve_whisperx_device
from transcribe_logic.whisperx_worker import (
    _maybe_assign_diarization_speakers,
    _to_segments,
)

_CACHE_LOCK = threading.RLock()
_ASR_CACHE: Dict[Tuple[str, str, str, str, str], Any] = {}
_ALIGN_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


def _load_whisperx():
    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError("Failed to import whisperx in runtime mode.") from exc
    return whisperx


def _get_asr_model(
    whisperx: Any,
    *,
    model: str,
    device: str,
    compute_type: str,
    language: str,
    vad_method: str,
) -> Any:
    key = (model, device, compute_type, language, vad_method)
    cached = _ASR_CACHE.get(key)
    if cached is not None:
        return cached

    kwargs: Dict[str, Any] = {
        "compute_type": compute_type,
        "language": language,
        "vad_method": vad_method,
    }
    try:
        asr_model = whisperx.load_model(model, device, **kwargs)
    except TypeError:
        # Backward compatibility for whisperx versions without vad_method.
        kwargs.pop("vad_method", None)
        asr_model = whisperx.load_model(model, device, **kwargs)

    _ASR_CACHE[key] = asr_model
    return asr_model


def _get_align_model(whisperx: Any, *, language_code: str, device: str) -> Tuple[Any, Any]:
    key = (language_code, device)
    cached = _ALIGN_CACHE.get(key)
    if cached is not None:
        return cached

    align_model, metadata = whisperx.load_align_model(
        language_code=language_code,
        device=device,
    )
    _ALIGN_CACHE[key] = (align_model, metadata)
    return align_model, metadata


def warmup_whisperx_runtime(
    *,
    model: str,
    language: str,
    device: str,
    compute_type: str,
    vad_method: str,
) -> None:
    """
    Optional server startup warmup to keep first request latency lower.
    """
    device = resolve_whisperx_device(device)
    whisperx = _load_whisperx()

    with _CACHE_LOCK:
        _get_asr_model(
            whisperx,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )
        _get_align_model(whisperx, language_code=language, device=device)


def whisperx_transcribe_inprocess(
    audio_path: str,
    *,
    model: str = "large-v3",
    language: str = "ru",
    device: str = "auto",
    compute_type: str = "int8",
    batch_size: int = 4,
    vad_method: str = "silero",
) -> List[Dict[str, Any]]:
    device = resolve_whisperx_device(device)
    whisperx = _load_whisperx()

    with _CACHE_LOCK:
        audio = whisperx.load_audio(audio_path)

        asr_model = _get_asr_model(
            whisperx,
            model=model,
            device=device,
            compute_type=compute_type,
            language=language,
            vad_method=vad_method,
        )
        result = asr_model.transcribe(audio, batch_size=batch_size, language=language)

        align_model, metadata = _get_align_model(
            whisperx,
            language_code=result["language"],
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        result = _maybe_assign_diarization_speakers(
            whisperx,
            result,
            audio_path=audio_path,
            device=device,
        )
        return _to_segments(result)
