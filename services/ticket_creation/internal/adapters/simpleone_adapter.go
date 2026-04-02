package adapters

import (
	"fmt"
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
	webhook *WebhookAdapter
}

func NewSimpleOneAdapter(cfg SimpleOneAdapterConfig) (*SimpleOneAdapter, error) {
	headers := map[string]string{}
	if cfg.BearerToken != "" {
		headers["Authorization"] = "Bearer " + cfg.BearerToken
	}

	webhook, err := NewWebhookAdapter(WebhookAdapterConfig{
		EndpointURL: cfg.EndpointURL,
		Headers:     headers,
		Timeout:     cfg.Timeout,
		SystemName:  "simpleone",
	})
	if err != nil {
		return nil, err
	}

	return &SimpleOneAdapter{webhook: webhook}, nil
}

func (a *SimpleOneAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	if a == nil || a.webhook == nil {
		return nil, fmt.Errorf("simpleone adapter is not configured")
	}
	if payload == nil {
		return nil, fmt.Errorf("ticket payload is required")
	}
	if payload.Draft == nil {
		return nil, fmt.Errorf("ticket payload draft is required")
	}

	return a.webhook.createTicketWithBody(buildSimpleOnePayload(payload), payload)
}

func (a *SimpleOneAdapter) GetTicket(externalID string) (*models.TicketCreated, error) {
	if a == nil || a.webhook == nil {
		return nil, fmt.Errorf("simpleone adapter is not configured")
	}
	return a.webhook.GetTicket(externalID)
}

func (a *SimpleOneAdapter) UpdateTicket(externalID string, update map[string]interface{}) error {
	if a == nil || a.webhook == nil {
		return fmt.Errorf("simpleone adapter is not configured")
	}
	return a.webhook.UpdateTicket(externalID, update)
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
