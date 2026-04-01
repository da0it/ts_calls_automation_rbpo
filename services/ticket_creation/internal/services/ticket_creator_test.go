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

	title := buildTicketTitle("portal_access")
	if title != "Обращение: portal_access" {
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
			IntentID:       "portal_access",
			Priority:       "high",
			SuggestedGroup: "technical_support",
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

func (failingTicketAdapter) CreateTicket(draft *models.TicketDraft) (*models.TicketCreated, error) {
	return nil, errors.New("stop after draft")
}

func (failingTicketAdapter) GetTicket(externalID string) (*models.TicketCreated, error) {
	return nil, errors.New("not implemented")
}

func (failingTicketAdapter) UpdateTicket(externalID string, update map[string]interface{}) error {
	return errors.New("not implemented")
}

var _ adapters.TicketSystemAdapter = failingTicketAdapter{}
