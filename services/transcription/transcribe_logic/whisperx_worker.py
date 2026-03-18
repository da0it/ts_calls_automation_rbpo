from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List


UNKNOWN_SPEAKER = "UNKNOWN"


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
                "speaker": UNKNOWN_SPEAKER,
                "text": text,
            }
        )

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--language", default="ru")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vad-method", default="silero", choices=["silero", "pyannote"])
    args = parser.parse_args()

    try:
        import whisperx
    except Exception as exc:
        raise RuntimeError(
            "Failed to import whisperx. Install it in the selected venv."
        ) from exc

    audio = whisperx.load_audio(args.audio)
    load_model_kwargs: Dict[str, Any] = {
        "compute_type": args.compute_type,
        "language": args.language,
        "vad_method": args.vad_method,
    }
    try:
        model = whisperx.load_model(args.model, args.device, **load_model_kwargs)
    except TypeError:
        # Backward compatibility for whisperx versions without vad_method arg.
        load_model_kwargs.pop("vad_method", None)
        model = whisperx.load_model(args.model, args.device, **load_model_kwargs)
    result = model.transcribe(audio, batch_size=args.batch_size, language=args.language)

    align_model, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=args.device,
    )
    result = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        args.device,
        return_char_alignments=False,
    )

    segments = _to_segments(result)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "language": result.get("language", "")}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
