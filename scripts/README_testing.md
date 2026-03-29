# Testing and Evaluation

This folder now contains reproducible scripts for the diploma chapter on testing and efficiency evaluation.

## 1. Functional and end-to-end checks

- Existing Go unit tests:

```bash
cd services/orchestrator && go test ./...
cd services/ticket_creation && go test ./...
```

- Batch end-to-end processing through the HTTP orchestrator API:

```bash
python3 scripts/evaluate_calls_folder.py \
  --base-url http://localhost:8000 \
  --username admin \
  --password 'YOUR_PASSWORD' \
  --input-dir /absolute/path/to/audio
```

This produces `results.csv`, `results.jsonl`, `summary.json` and can be used as a functional smoke check for the full pipeline.

## 2. Classification quality

After manual labeling of routing results, run:

```bash
python3 scripts/evaluate_routing_csv.py \
  --csv /absolute/path/to/secure_labeling_dataset.csv
```

The script computes `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1` for `intent`, `group` and `priority`.

## 3. Transcription quality

Prepare a CSV with two columns: a reference transcript and the ASR hypothesis. Then run:

```bash
python3 scripts/evaluate_transcription_wer.py \
  --csv /absolute/path/to/transcription_eval.csv \
  --ref-col reference_text \
  --hyp-col hypothesis_text
```

The script computes:

- `WER`
- `CER`
- total substitutions, deletions and insertions
- perfect-match rate
- a JSON report `transcription_metrics.json`

## 4. A/B testing

Prepare a CSV with at least these columns:

- `group` (`A` or `B`)
- `classification_time_sec`
- `is_error`
- `operator_load_pct`

Run:

```bash
python3 scripts/evaluate_ab_test.py \
  --csv /absolute/path/to/ab_results.csv
```

The report contains per-group averages plus reductions between control (`A`) and treatment (`B`) for time, error rate and operator load.

## 5. Load testing

Use k6 against the authenticated `/api/v1/process-call` endpoint:

```bash
BASE_URL=http://localhost:8000 \
USERNAME=admin \
PASSWORD='YOUR_PASSWORD' \
AUDIO_FILES=/absolute/path/to/call1.wav,/absolute/path/to/call2.wav \
k6 run scripts/load_test_process_call.js
```

The default scenario ramps load up, keeps a short soak period and checks:

- failed request rate
- average latency
- 95th percentile latency
- response integrity (`transcript` and `routing` present)

You can override stage parameters through environment variables such as `TARGET_VUS_1`, `TARGET_VUS_2`, `SOAK`, `RAMP_DOWN`.

## 6. Script self-tests

To verify the evaluation scripts themselves:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
