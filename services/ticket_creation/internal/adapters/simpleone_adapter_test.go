package adapters

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ticket_module/internal/models"
)

func TestSimpleOneAdapterCreateTicketSendsPayloadAndBearer(t *testing.T) {
	t.Parallel()

	var gotAuth string
	var gotPayload models.TicketSystemPayload

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}
		gotAuth = r.Header.Get("Authorization")
		if err := json.NewDecoder(r.Body).Decode(&gotPayload); err != nil {
			t.Fatalf("decode payload: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"data":{"id":"INC000123","record_url":"https://simpleone.example/record/INC000123"}}`))
	}))
	defer server.Close()

	adapter, err := NewSimpleOneAdapter(SimpleOneAdapterConfig{
		EndpointURL: server.URL,
		BearerToken: "secret-token",
		Timeout:     5 * time.Second,
	})
	if err != nil {
		t.Fatalf("new adapter: %v", err)
	}

	created, err := adapter.CreateTicket(sampleTicketPayload())
	if err != nil {
		t.Fatalf("create ticket: %v", err)
	}

	if gotAuth != "Bearer secret-token" {
		t.Fatalf("unexpected auth header: %q", gotAuth)
	}
	if gotPayload.Service.Source != "ts_calls_automation" {
		t.Fatalf("unexpected service source: %+v", gotPayload.Service)
	}
	if gotPayload.Request.Transcript.CallID != "call-123" {
		t.Fatalf("unexpected call id in payload: %+v", gotPayload.Request.Transcript)
	}
	if gotPayload.Draft == nil || !strings.Contains(gotPayload.Draft.Description, "Проблема") {
		t.Fatalf("unexpected draft in payload: %+v", gotPayload.Draft)
	}
	if created.ExternalID != "INC000123" {
		t.Fatalf("unexpected external id: %+v", created)
	}
	if created.URL != "https://simpleone.example/record/INC000123" {
		t.Fatalf("unexpected url: %+v", created)
	}
	if created.System != "simpleone" {
		t.Fatalf("unexpected system: %+v", created)
	}
}

func TestSimpleOneAdapterCreateTicketFallsBackToAckID(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"status":"accepted"}`))
	}))
	defer server.Close()

	adapter, err := NewSimpleOneAdapter(SimpleOneAdapterConfig{
		EndpointURL: server.URL,
		Timeout:     5 * time.Second,
	})
	if err != nil {
		t.Fatalf("new adapter: %v", err)
	}

	created, err := adapter.CreateTicket(sampleTicketPayload())
	if err != nil {
		t.Fatalf("create ticket: %v", err)
	}

	if created.ExternalID != "simpleone-ack-call-123" {
		t.Fatalf("unexpected fallback external id: %+v", created)
	}
}

func sampleTicketPayload() *models.TicketSystemPayload {
	return &models.TicketSystemPayload{
		Service: models.TicketServiceMetadata{
			Source:        "ts_calls_automation",
			Component:     "ticket_creation",
			SchemaVersion: "v1",
			SentAt:        time.Unix(1712345678, 0).UTC(),
			CallID:        "call-123",
			IntentID:      "portal_access",
			Priority:      "high",
		},
		Request: models.CreateTicketRequest{
			Transcript: models.TranscriptData{
				CallID: "call-123",
				Segments: []models.Segment{
					{Role: "client", Text: "Не могу войти в личный кабинет."},
				},
				Metadata: map[string]interface{}{
					"source_file": "call.wav",
				},
			},
			Routing: models.RoutingData{
				IntentID:         "portal_access",
				IntentConfidence: 0.97,
				Priority:         "high",
				SuggestedGroup:   "tech_support",
			},
			Entities: &models.Entities{
				Phones: []models.ExtractedEntity{{Value: "+79991234567"}},
			},
			AudioURL: "https://storage.example/call.wav",
		},
		Summary: &models.TicketSummary{
			Description: "Проблема: Клиент не может войти в личный кабинет.",
		},
		Draft: &models.TicketDraft{
			Title:        "Обращение: portal_access",
			Description:  "Проблема: Клиент не может войти в личный кабинет.",
			Priority:     "high",
			AssigneeType: "group",
			AssigneeID:   "tech_support",
			Tags:         []string{"portal_access", "urgent"},
			CallID:       "call-123",
			AudioURL:     "https://storage.example/call.wav",
			IntentID:     "portal_access",
		},
	}
}
