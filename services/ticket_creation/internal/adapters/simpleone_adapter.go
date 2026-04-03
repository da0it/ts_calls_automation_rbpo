package adapters

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"ticket_module/internal/models"
)

type SimpleOneAdapterConfig struct {
	EndpointURL string
	BearerToken string
	Timeout     time.Duration
}

type SimpleOneAdapter struct {
	endpointURL string
	bearerToken string
	httpClient  *http.Client
}

func NewSimpleOneAdapter(cfg SimpleOneAdapterConfig) (*SimpleOneAdapter, error) {
	endpointURL := strings.TrimSpace(cfg.EndpointURL)
	if endpointURL == "" {
		return nil, fmt.Errorf("simpleone endpoint URL is required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 30 * time.Second
	}
	return &SimpleOneAdapter{
		endpointURL: endpointURL,
		bearerToken: strings.TrimSpace(cfg.BearerToken),
		httpClient:  &http.Client{Timeout: cfg.Timeout},
	}, nil
}

func (a *SimpleOneAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	if a == nil {
		return nil, fmt.Errorf("simpleone adapter is not configured")
	}
	if payload == nil {
		return nil, fmt.Errorf("ticket payload is required")
	}
	if payload.Draft == nil {
		return nil, fmt.Errorf("ticket payload draft is required")
	}

	body, err := json.Marshal(buildSimpleOnePayload(payload))
	if err != nil {
		return nil, fmt.Errorf("marshal simpleone payload: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, a.endpointURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create simpleone request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "ts-calls-automation/ticket-service")
	if a.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+a.bearerToken)
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send simpleone request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read simpleone response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("simpleone response %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	return buildSimpleOneCreated(respBody, payload), nil
}

func buildSimpleOnePayload(payload *models.TicketSystemPayload) map[string]interface{} {
	draft := payload.Draft
	request := payload.Request
	summary := payload.Summary

	body := map[string]interface{}{
		"title":               draft.Title,
		"description":         draft.Description,
		"priority":            draft.Priority,
		"assignee_type":       draft.AssigneeType,
		"assignee_id":         draft.AssigneeID,
		"tags":                draft.Tags,
		"call_id":             draft.CallID,
		"audio_url":           draft.AudioURL,
		"transcript_url":      draft.TranscriptURL,
		"intent_id":           draft.IntentID,
		"intent_confidence":   draft.IntentConfidence,
		"custom_fields":       draft.CustomFields,
		"problem_summary":     collapseWhitespace(simpleOneProblemSummary(summary, draft)),
		"transcript_text":     simpleOneTranscriptText(request.Transcript.Segments),
		"transcript_segments": request.Transcript.Segments,
		"entities":            draft.Entities,
		"service":             payload.Service,
		"request":             request,
		"summary":             summary,
		"draft":               draft,
	}

	if sourceFile := simpleOneSourceFile(request.Transcript.Metadata); sourceFile != "" {
		body["source_file"] = sourceFile
	}

	return body
}

func simpleOneProblemSummary(summary *models.TicketSummary, draft *models.TicketDraft) string {
	if summary != nil {
		if value := strings.TrimSpace(summary.Description); value != "" {
			return value
		}
	}
	if draft != nil {
		return strings.TrimSpace(draft.Description)
	}
	return ""
}

func simpleOneTranscriptText(segments []models.Segment) string {
	if len(segments) == 0 {
		return ""
	}

	lines := make([]string, 0, len(segments))
	for _, segment := range segments {
		text := collapseWhitespace(segment.Text)
		if text == "" {
			continue
		}

		label := collapseWhitespace(segment.Role)
		if label == "" {
			label = collapseWhitespace(segment.Speaker)
		}
		if label != "" {
			lines = append(lines, label+": "+text)
		} else {
			lines = append(lines, text)
		}
	}

	return strings.Join(lines, "\n")
}

func simpleOneSourceFile(metadata map[string]interface{}) string {
	if len(metadata) == 0 {
		return ""
	}
	if raw, ok := metadata["source_file"]; ok {
		if value, ok := raw.(string); ok {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func collapseWhitespace(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}

func buildSimpleOneCreated(respBody []byte, payload *models.TicketSystemPayload) *models.TicketCreated {
	created := &models.TicketCreated{
		ExternalID: simpleOneAckID(payload),
		System:     "simpleone",
		CreatedAt:  time.Now().UTC(),
	}
	if strings.TrimSpace(string(respBody)) == "" {
		return created
	}

	var decoded map[string]interface{}
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return created
	}

	data, _ := decoded["data"].(map[string]interface{})
	if id, ok := data["id"].(string); ok && strings.TrimSpace(id) != "" {
		created.ExternalID = strings.TrimSpace(id)
	}
	if url, ok := data["record_url"].(string); ok && strings.TrimSpace(url) != "" {
		created.URL = strings.TrimSpace(url)
	}
	return created
}

func simpleOneAckID(payload *models.TicketSystemPayload) string {
	if payload != nil && payload.Service.CallID != "" {
		return "simpleone-ack-" + payload.Service.CallID
	}
	return "simpleone-ack"
}
