# Testing

For the diploma, it is enough to refer to 3 main test files:

1. `tests/run_functional_tests.py`
   Functional checks of the external HTTP API:
   login, validation of input audio, successful processing, ticket creation, notification, audit log, role separation.

2. `services/orchestrator/tests/orchestrator_integration_test.go`
   Integration tests of the internal service chain:
   transcription -> routing -> entity extraction -> ticket creation -> notification,
   and stopping the pipeline when manual routing review is required.

3. `tests/evaluate_ab_test.py`
   Comparative test of the subsystem and manual operator classification:
   agreement by intent, group and priority, time reduction, operator load reduction.

Typical commands:

```bash
python3 tests/run_functional_tests.py \
  --base-url http://localhost:8000 \
  --admin-username admin \
  --admin-password 'YOUR_PASSWORD' \
  --audio /absolute/path/to/sample.wav
```

```bash
cd services/orchestrator && go test ./...
```

```bash
python3 tests/evaluate_ab_test.py \
  --csv /absolute/path/to/secure_labeling_dataset.csv
```
