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

If you already have a labeled CSV with file paths, intents and spam labels, you can pass it into the batch evaluation:

```bash
python3 scripts/evaluate_calls_folder.py \
  --base-url http://localhost:8000 \
  --username admin \
  --password 'YOUR_PASSWORD' \
  --input-dir /absolute/path/to/audio \
  --labels-csv /absolute/path/to/real_mixed_eval.csv \
  --labels-path-col source_file \
  --labels-intent-col call_purpose \
  --labels-spam-col manual_is_spam
```

The resulting `results.csv` will include:

- `status`
- `spam_status`
- `spam_predicted_label`
- `spam_confidence`
- `manual_is_spam`
- `true_intent`
- `pred_intent`

## 2. Classification quality

After manual labeling of routing results, run:

```bash
python3 scripts/evaluate_routing_csv.py \
  --csv /absolute/path/to/secure_labeling_dataset.csv
```

The script computes `accuracy`, `macro_precision`, `macro_recall`, `macro_f1`, `weighted_f1` for `intent`, `group` and `priority`.

If your CSV contains only intent labels such as `call_purpose` plus model output `pred_intent`, that is also supported now:

```bash
python3 scripts/evaluate_routing_csv.py \
  --csv /absolute/path/to/results.csv \
  --intent-true-col true_intent \
  --intent-pred-col pred_intent
```

When `group` or `priority` columns are missing, the script skips those metrics automatically.

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

### 5.1 Advanced load testing with pipeline-stage metrics

For diploma-grade measurements, use the advanced script:

```bash
BASE_URL=http://localhost:8000 \
USERNAME=admin \
PASSWORD='YOUR_PASSWORD' \
AUDIO_FILES=/absolute/path/to/call1.wav,/absolute/path/to/call2.wav \
WORKLOAD_MODEL=closed \
SUMMARY_JSON=load_test_summary.json \
SUMMARY_TEXT=load_test_summary.txt \
k6 run scripts/load_test_process_call_advanced.js
```

The advanced scenario supports:

- `closed` workload model via `ramping-vus` for concurrent-user simulation
- `open` workload model via `ramping-arrival-rate` for arrival-rate studies
- warm-up, ramp-up, steady-state soak and ramp-down phases
- pass/fail thresholds for HTTP error rate, `p95`/`p99` latency and payload integrity
- custom bottleneck metrics extracted from `processing_time` in the orchestrator response:
  - transcription
  - routing
  - entity extraction
  - ticket creation
  - notification
- summary export to JSON and plain text

Example for the open workload model:

```bash
BASE_URL=http://localhost:8000 \
USERNAME=admin \
PASSWORD='YOUR_PASSWORD' \
AUDIO_FILES=/absolute/path/to/call1.wav,/absolute/path/to/call2.wav \
WORKLOAD_MODEL=open \
ARRIVAL_TIME_UNIT=1m \
START_RATE=1 \
TARGET_RATE_1=2 \
TARGET_RATE_2=4 \
PRE_ALLOCATED_VUS=8 \
MAX_VUS=32 \
k6 run scripts/load_test_process_call_advanced.js
```

Useful environment overrides:

- `WORKLOAD_MODEL=closed|open`
- `WARMUP_DURATION`, `RAMP_UP_1`, `RAMP_UP_2`, `SOAK`, `RAMP_DOWN`
- `START_VUS`, `WARMUP_VUS`, `TARGET_VUS_1`, `TARGET_VUS_2`
- `ARRIVAL_TIME_UNIT`, `START_RATE`, `WARMUP_RATE`, `TARGET_RATE_1`, `TARGET_RATE_2`, `PRE_ALLOCATED_VUS`, `MAX_VUS`
- `REQUEST_TIMEOUT`
- `HTTP_P95_MS`, `HTTP_P99_MS`
- `PIPELINE_P95_SEC`, `PIPELINE_P99_SEC`
- `MAX_HTTP_ERROR_RATE`, `MAX_HTTP_5XX_RATE`, `MAX_INCOMPLETE_RATE`

### 5.2 Methodology notes and references

The advanced script is based on standard performance-testing practices:

- phased workload profile: warm-up, ramp-up, steady-state soak, ramp-down
- percentile-based latency control (`p95`, `p99`) instead of averages only
- explicit error-rate thresholds
- separation of external HTTP latency from internal pipeline-stage timing
- support for both closed and open workload models, depending on whether the target study is concurrent-user behavior or fixed arrival rate

Suggested references for the thesis:

1. Raj Jain, *The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling*. Wiley, 1991.  
   Publisher / bibliographic pages:  
   [WashU profile](https://profiles.wustl.edu/en/publications/the-art-of-computer-systems-performance-analysis-techniques-for-e)  
   [Open Library](https://openlibrary.org/books/OL1884550M/The_art_of_computer_systems_performance_analysis)

2. Ian Molyneaux, *The Art of Application Performance Testing*. O'Reilly Media, 2009; 2nd ed., 2014.  
   Bibliographic pages:  
   [O'Reilly](https://www.oreilly.com/library/view/the-art-of/9780596155858/)  
   [Google Books](https://books.google.com/books/about/The_Art_of_Application_Performance_Testi.html?id=DccaCe9WzeoC)

3. Grafana k6 documentation, *Open and closed models*.  
   This source is useful for justifying the choice between concurrent-user and arrival-rate workload generation:  
   [https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/](https://grafana.com/docs/k6/latest/using-k6/scenarios/concepts/open-vs-closed/)

4. Grafana k6 documentation, *Thresholds*.  
   This source supports the use of formal pass/fail criteria for percentile latency and error rate:  
   [https://grafana.com/docs/k6/latest/using-k6/thresholds/](https://grafana.com/docs/k6/latest/using-k6/thresholds/)

5. Google SRE Book, *Service Level Objectives*.  
   This source supports the use of high-percentile latency metrics instead of average latency alone:  
   [https://sre.google/sre-book/service-level-objectives/](https://sre.google/sre-book/service-level-objectives/)

## 6. Script self-tests

To verify the evaluation scripts themselves:

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```
