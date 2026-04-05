package services

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"ticket_module/internal/adapters"
	"ticket_module/internal/clients"
	"ticket_module/internal/models"
)

func TestBuildTicketTitleUsesIntent(t *testing.T) {
	t.Parallel()

	title := buildTicketTitle("misc.triage")
	if title != "Обращение: misc.triage" {
		t.Fatalf("unexpected title: %q", title)
	}
}

func TestComposeTicketDescriptionUsesSummary(t *testing.T) {
	t.Parallel()

	description := composeTicketDescription(&models.TicketSummary{
		Description: "Проблема: Клиент не может войти в личный кабинет после смены пароля.",
	})

	if !strings.Contains(description, "Клиент не может войти") {
		t.Fatalf("description should include summary body: %q", description)
	}
}

func TestAppendEntityDetailsAddsExtraInfo(t *testing.T) {
	t.Parallel()

	description := appendEntityDetails("Проблема: Не удается войти в кабинет.", &models.Entities{
		Persons: []models.ExtractedEntity{{Value: "Иван Петров"}},
		Phones:  []models.ExtractedEntity{{Value: "+79991234567"}},
	})

	if !strings.Contains(description, "Доп. информация:") {
		t.Fatalf("description should include extra info block: %q", description)
	}
	if !strings.Contains(description, "Иван Петров") {
		t.Fatalf("description should include person name: %q", description)
	}
}

func TestCreateTicketSkipsPythonNERWhenEntitiesAlreadyProvided(t *testing.T) {
	t.Parallel()

	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, "unexpected ner call", http.StatusInternalServerError)
	}))
	defer server.Close()

	service := NewTicketCreatorService(
		clients.NewPythonClient(server.URL),
		staticSummarizer{},
		failingTicketAdapter{},
		nil,
		false,
	)

	_, err := service.CreateTicket(&models.CreateTicketRequest{
		Transcript: models.TranscriptData{
			CallID:   "call-1",
			Segments: []models.Segment{{Role: "client", Text: "Не могу войти в портал."}},
		},
		Routing: models.RoutingData{
			IntentID:       "misc.triage",
			Priority:       "high",
			SuggestedGroup: "support",
		},
		Entities: &models.Entities{
			Phones: []models.ExtractedEntity{{Value: "+79991234567"}},
		},
	})
	if err == nil {
		t.Fatal("expected adapter error")
	}
	if hits.Load() != 0 {
		t.Fatalf("python NER should not be called when entities are provided, got %d hits", hits.Load())
	}
}

type staticSummarizer struct{}

func (staticSummarizer) GenerateSummary(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
	return &models.TicketSummary{
		Description: "Проблема: Тестовое описание.",
	}, nil
}

type failingTicketAdapter struct{}

func (failingTicketAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	return nil, errors.New("stop after draft")
}

type capturingTicketAdapter struct {
	payload *models.TicketSystemPayload
}

func (a *capturingTicketAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	a.payload = payload
	return &models.TicketCreated{
		TicketID:   "ticket-1",
		ExternalID: "EXT-1",
		URL:        "http://ticket.local/ticket-1",
		System:     "simpleone",
	}, nil
}

func TestCreateTicketBuildsDraftFromRoutingSummaryAndEntities(t *testing.T) {
	t.Parallel()

	adapter := &capturingTicketAdapter{}
	service := NewTicketCreatorService(
		nil,
		staticSummarizer{},
		adapter,
		nil,
		false,
	)

	created, err := service.CreateTicket(&models.CreateTicketRequest{
		Transcript: models.TranscriptData{
			CallID: "call-2",
			Segments: []models.Segment{
				{Speaker: "spk_0", Text: "Не приходит код подтверждения."},
			},
		},
		Routing: models.RoutingData{
			IntentID:         "portal_access",
			IntentConfidence: 0.93,
			Priority:         "high",
			SuggestedGroup:   "support",
		},
		Entities: &models.Entities{
			Phones: []models.ExtractedEntity{{Value: "+79991234567"}},
		},
	})
	if err != nil {
		t.Fatalf("CreateTicket returned error: %v", err)
	}
	if created == nil || created.System != "simpleone" {
		t.Fatalf("unexpected created ticket: %#v", created)
	}
	if adapter.payload == nil {
		t.Fatal("ticket adapter did not receive payload")
	}
	if adapter.payload.Draft == nil {
		t.Fatal("draft is missing in payload")
	}
	if adapter.payload.Draft.Title != "Обращение: portal_access" {
		t.Fatalf("unexpected draft title: %q", adapter.payload.Draft.Title)
	}
	if adapter.payload.Draft.AssigneeID != "support" {
		t.Fatalf("unexpected assignee: %q", adapter.payload.Draft.AssigneeID)
	}
	if !strings.Contains(adapter.payload.Draft.Description, "Тестовое описание") {
		t.Fatalf("unexpected draft description: %q", adapter.payload.Draft.Description)
	}
	if adapter.payload.Request.Routing.IntentID != "portal_access" {
		t.Fatalf("unexpected routing intent in payload: %q", adapter.payload.Request.Routing.IntentID)
	}
	if adapter.payload.Request.Entities == nil || len(adapter.payload.Request.Entities.Phones) != 1 {
		t.Fatalf("entities were not passed to payload: %#v", adapter.payload.Request.Entities)
	}
}

var _ adapters.TicketSystemAdapter = failingTicketAdapter{}
var _ adapters.TicketSystemAdapter = (*capturingTicketAdapter)(nil)
