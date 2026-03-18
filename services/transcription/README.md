# transcribe_module

## Backend

Only WhisperX is supported in this service.

## WhisperX requirements

`WHISPERX_VENV_PYTHON` must point to a python with installed `whisperx`:

```bash
/Users/dmitrii/whisperx_venv/bin/pip install whisperx
```

The service currently uses WhisperX only for ASR plus timestamp alignment.
Speaker diarization is disabled in the code path.
