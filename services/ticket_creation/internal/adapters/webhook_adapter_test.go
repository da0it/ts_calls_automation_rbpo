package adapters

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"ticket_module/internal/models"
)

func TestWebhookAdapterUsesConfigurableHeadersAndResponsePaths(t *testing.T) {
	t.Parallel()

	var gotAPIKey string
	var gotPayload models.TicketSystemPayload

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAPIKey = r.Header.Get("X-Api-Key")
		if err := json.NewDecoder(r.Body).Decode(&gotPayload); err != nil {
			t.Fatalf("decode payload: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"result":{"ticket":{"id":"HD-77","links":[{"href":"https://helpdesk.example/tickets/HD-77"}]}}}`))
	}))
	defer server.Close()

	adapter, err := NewWebhookAdapter(WebhookAdapterConfig{
		EndpointURL:    server.URL,
		Headers:        map[string]string{"X-Api-Key": "abc123"},
		Timeout:        5 * time.Second,
		SystemName:     "helpdesk",
		ExternalIDPath: "result.ticket.id",
		URLPath:        "result.ticket.links.0.href",
	})
	if err != nil {
		t.Fatalf("new adapter: %v", err)
	}

	created, err := adapter.CreateTicket(sampleTicketPayload())
	if err != nil {
		t.Fatalf("create ticket: %v", err)
	}

	if gotAPIKey != "abc123" {
		t.Fatalf("unexpected API key header: %q", gotAPIKey)
	}
	if gotPayload.Request.Transcript.CallID != "call-123" {
		t.Fatalf("unexpected payload: %+v", gotPayload)
	}
	if created.ExternalID != "HD-77" {
		t.Fatalf("unexpected external id: %+v", created)
	}
	if created.URL != "https://helpdesk.example/tickets/HD-77" {
		t.Fatalf("unexpected url: %+v", created)
	}
	if created.System != "helpdesk" {
		t.Fatalf("unexpected system: %+v", created)
	}
}

func TestParseHeadersJSONAcceptsScalarValues(t *testing.T) {
	t.Parallel()

	headers, err := ParseHeadersJSON(`{"X-Api-Key":"secret","X-Retry":2,"X-Enabled":true}`)
	if err != nil {
		t.Fatalf("parse headers: %v", err)
	}

	if headers["X-Api-Key"] != "secret" {
		t.Fatalf("unexpected api key: %#v", headers)
	}
	if headers["X-Retry"] != "2" {
		t.Fatalf("unexpected retry header: %#v", headers)
	}
	if headers["X-Enabled"] != "true" {
		t.Fatalf("unexpected enabled header: %#v", headers)
	}
}
