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
	var gotPayload map[string]interface{}

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
	if gotPayload["title"] != "Обращение: portal_access" {
		t.Fatalf("unexpected title in payload: %#v", gotPayload["title"])
	}
	if gotPayload["problem_summary"] != "Проблема: Клиент не может войти в личный кабинет." {
		t.Fatalf("unexpected problem summary in payload: %#v", gotPayload["problem_summary"])
	}
	if transcriptText, _ := gotPayload["transcript_text"].(string); !strings.Contains(transcriptText, "client: Не могу войти в личный кабинет.") {
		t.Fatalf("unexpected transcript_text in payload: %#v", gotPayload["transcript_text"])
	}
	request, _ := gotPayload["request"].(map[string]interface{})
	transcript, _ := request["transcript"].(map[string]interface{})
	if transcript["call_id"] != "call-123" {
		t.Fatalf("unexpected call id in nested request payload: %#v", gotPayload["request"])
	}
	if gotPayload["source_file"] != "call.wav" {
		t.Fatalf("unexpected source_file in payload: %#v", gotPayload["source_file"])
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
