from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

try:
    from transcribe_logic.config import get_whisperx_settings
except ImportError:
    from config import get_whisperx_settings

UNKNOWN_SPEAKER = ""

def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

def _maybe_assign_diarization_speakers(
    whisperx: Any,
    result: Dict[str, Any],
    *,
    audio_path: str,
    device: str,
) -> Dict[str, Any]:
    if not _env_bool("WHISPERX_ENABLE_DIARIZATION", False):
        return result

    hf_token = (
        os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_TOKEN", "").strip()
        or os.getenv("HF_HUB_TOKEN", "").strip()
    )
    if not hf_token:
        return result

    diarization_pipeline = getattr(whisperx, "DiarizationPipeline", None)
    assign_word_speakers = getattr(whisperx, "assign_word_speakers", None)
    if diarization_pipeline is None or assign_word_speakers is None:
        return result

    min_speakers = max(1, int(os.getenv("WHISPERX_MIN_SPEAKERS", "2")))
    max_speakers = max(min_speakers, int(os.getenv("WHISPERX_MAX_SPEAKERS", str(min_speakers))))

    diarize_model = diarization_pipeline(
        use_auth_token=hf_token,
        device=device,
    )
    diarize_segments = diarize_model(
        audio_path,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return assign_word_speakers(diarize_segments, result)

def _to_segments(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for seg in result.get("segments", []):
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue
        out.append(
            {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "speaker": str(seg.get("speaker", UNKNOWN_SPEAKER) or "").strip(),
                "text": text,
            }
        )

    return out

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()
    settings = get_whisperx_settings()
    device = settings["device"]

    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError(
            "Failed to import whisperx. Install it in the selected venv."
        ) from exc

    audio = whisperx.load_audio(args.audio)
    load_model_kwargs: Dict[str, Any] = {
        "compute_type": settings["compute_type"],
        "language": settings["language"],
        "vad_method": settings["vad_method"],
    }
    try:
        model = whisperx.load_model(settings["model"], device, **load_model_kwargs)
    except TypeError:
        # Backward compatibility for whisperx versions without vad_method arg.
        load_model_kwargs.pop("vad_method", None)
        model = whisperx.load_model(settings["model"], device, **load_model_kwargs)
    result = model.transcribe(audio, batch_size=settings["batch_size"], language=settings["language"])

    align_model, metadata = whisperx.load_align_model(
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
        audio_path=args.audio,
        device=device,
    )

    segments = _to_segments(result)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "language": result.get("language", "")}, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
