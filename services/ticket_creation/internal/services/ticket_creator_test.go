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
		Description:       "Клиент не может войти в личный кабинет после смены пароля.",
		KeyPoints:         []string{"Сброс пароля не помог", "Ошибка сохраняется"},
		SuggestedSolution: "Проверить статус учётной записи и принудительно сбросить пароль.",
	})

	if !strings.Contains(description, "Клиент не может войти") {
		t.Fatalf("description should include summary body: %q", description)
	}
	if !strings.Contains(description, "Ключевые моменты:") {
		t.Fatalf("description should include key points: %q", description)
	}
}
