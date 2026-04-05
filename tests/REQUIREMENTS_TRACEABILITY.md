# Requirements Traceability

This file maps the diploma functional requirements to the current automated tests.

## Top-level functional requirements

1. Access to incoming calls, secure intake, persistence
   - `tests/run_functional_tests.py`: `T1`, `T3`, `T4`, `T5`, `T21`, `T22`, `T23`, `T24`, `T30`

2. Automatic speech transcription with preprocessing
   - `tests/run_functional_tests.py`: `T6+`, `T31`
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationFullPipeline`

3. Automatic intent, confidence, priority and routing
   - `tests/run_functional_tests.py`: `T6+`, `T32`
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationFullPipeline`
   - `tests/test_entity_extraction_normalization.py`

4. Automatic registration in linked ticket system
   - `tests/run_functional_tests.py`: `T20`, `T33`
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationFullPipeline`
   - `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`
   - `services/ticket_creation/internal/services/ticket_creator_test.go`

5. Notification of responsible side
   - `tests/run_functional_tests.py`: `T25`, `T34`
   - `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`

Note:
In the current diploma version notifications are delegated to the linked ticket system. Coverage for requirement 5 is therefore checked as successful handoff of the assigned ticket to the external system.

## Detailed requirements by subsystem

### 2.3.2.1 Request intake and verification

1. Accept `mp3`, `ogg`, `wav`
   - `tests/run_functional_tests.py`: successful `T6+` runs for supplied `.mp3`, `.ogg`, `.wav`
   - Full format coverage requires passing all three formats to the script.

2. User authentication
   - `tests/run_functional_tests.py`: `T1`, `T2`, `T3`

3. Role-based access
   - `tests/run_functional_tests.py`: `T23`, `T24`, `T36`

4. Input validation
   - `tests/run_functional_tests.py`: `T4`, `T5`

5. Validated processing request
   - `tests/run_functional_tests.py`: successful `T6+`

### 2.3.2.2 Transcription

1. Speech recognition from audio
   - `tests/run_functional_tests.py`: successful `T6+`
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationFullPipeline`

2. Structured transcript
   - `tests/run_functional_tests.py`: `T6+` validates `segments`, `start`, `end`, `speaker`, `text`

3. Saved transcription status for control and audit
   - `tests/run_functional_tests.py`: `T22`, `T31`

### 2.3.2.3 Classification and entity extraction

1. Intent detection
   - `tests/run_functional_tests.py`: `T6+`, `T32`

2. Confidence value
   - `tests/run_functional_tests.py`: `T6+`, `T32`

3. Priority
   - `tests/run_functional_tests.py`: `T6+`, `T32`

4. Routing to target group
   - `tests/run_functional_tests.py`: `T6+`, `T32`

5. Named entity extraction
   - `tests/run_functional_tests.py`: `T6+`
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationFullPipeline`

6. Entity normalization and deduplication
   - `tests/test_entity_extraction_normalization.py`

7. Structured parameters for ticket creation
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationFullPipeline`
   - `services/ticket_creation/internal/services/ticket_creator_test.go`: `TestCreateTicketBuildsDraftFromRoutingSummaryAndEntities`

### 2.3.2.4 Ticket analysis and registration

1. Ticket title generation
   - `services/ticket_creation/internal/services/ticket_creator_test.go`: `TestBuildTicketTitleUsesIntent`
   - `services/ticket_creation/internal/services/ticket_creator_test.go`: `TestCreateTicketBuildsDraftFromRoutingSummaryAndEntities`

2. Summary/description generation
   - `services/ticket_creation/internal/services/ticket_creator_test.go`: `TestComposeTicketDescriptionUsesSummary`
   - `services/ticket_creation/internal/services/summarizer_test.go`: `TestGenerateSummaryWithOllama`

3. Fill ticket fields from classification and entities
   - `services/ticket_creation/internal/services/ticket_creator_test.go`: `TestCreateTicketBuildsDraftFromRoutingSummaryAndEntities`
   - `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`: `TestSimpleOneAdapterCreateTicketSendsPayloadAndBearer`

4. Create ticket in external system
   - `tests/run_functional_tests.py`: `T20`, `T33`
   - `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`

5. Receive and save ticket ID and link
   - `tests/run_functional_tests.py`: `T20`
   - `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`

6. Register result even when one stage fails
   - `services/orchestrator/tests/orchestrator_integration_test.go`: `TestProcessCallIntegrationContinuesWhenEntityExtractionFails`

### 2.3.2.5 Notification of responsible parties

1. Form notification from created ticket
2. Support delivery to notification channels
3. Store per-channel delivery result
4. Store delivery errors

For the current diploma implementation these checks are delegated to the linked ticket system and covered as successful ticket handoff with assignee/group:
- `tests/run_functional_tests.py`: `T25`, `T34`
- `services/ticket_creation/internal/adapters/simpleone_adapter_test.go`: `TestSimpleOneAdapterCreateTicketSendsPayloadAndBearer`

### 2.3.2.6 Storage and audit

1. Save processing report for each call
   - `tests/run_functional_tests.py`: `T21`, `T35`

2. Report contains statuses, intent, entities, ticket data
   - `tests/run_functional_tests.py`: `T21`, `T22`, `T35`

3. Subsequent audit
   - `tests/run_functional_tests.py`: `T22`

4. Data accumulation for quality analysis
   - `tests/run_functional_tests.py`: `T21`, `T22`
   - `services/orchestrator/internal/services/audit_service.go`
   - `services/orchestrator/internal/services/call_queue_service.go`

### 2.3.2.7 Security

1. Role-based access separation
   - `tests/run_functional_tests.py`: `T23`, `T24`, `T36`

2. Local processing without external cloud transfer
   - `services/ticket_creation/internal/services/summarizer_test.go`: `TestGenerateSummaryRejectsExternalOllamaURL`

3. Logging of significant operations
   - `tests/run_functional_tests.py`: `T22`, `T35`
