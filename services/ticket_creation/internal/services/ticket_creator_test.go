package services

import (
	"strings"
	"testing"

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
