# transcribe_module

## Backend

Only WhisperX is supported in this service.

## WhisperX requirements

`WHISPERX_VENV_PYTHON` must point to a python with installed `whisperx`:

```bash
/Users/dmitrii/whisperx_venv/bin/pip install whisperx
```

The service uses WhisperX for ASR plus timestamp alignment.
Speaker diarization is optional and requires both `WHISPERX_ENABLE_DIARIZATION=1` and a valid `HF_TOKEN`.
If diarization is unavailable, the service returns transcript segments without speaker labels.
