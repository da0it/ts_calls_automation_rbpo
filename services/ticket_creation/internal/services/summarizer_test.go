package services

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"ticket_module/internal/models"
)

func TestGenerateSummaryWithOllama(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Fatalf("unexpected method: %s", r.Method)
		}

		var req ollamaRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "qwen2.5:7b" {
			t.Fatalf("unexpected model: %s", req.Model)
		}
		if req.Format != "json" {
			t.Fatalf("expected json format, got %q", req.Format)
		}
		if !strings.Contains(req.Prompt, "delivery_delay") {
			t.Fatalf("prompt should contain intent, got %q", req.Prompt)
		}

		resp := ollamaResponse{
			Response: `{"title":"Задержка доставки заказа","description":"Клиент сообщает, что заказ не доставлен в обещанный срок.","key_points":["Заказ не приехал вовремя","Клиент просит проверить статус"],"suggested_solution":"Проверить статус у логистики и сообщить новый срок доставки.","urgency_reason":"Клиент ждёт заказ сегодня."}`,
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	summarizer := NewLLMSummarizer(SummarizerConfig{
		Provider:          "ollama",
		OllamaBaseURL:     server.URL,
		OllamaModel:       "qwen2.5:7b",
		OllamaTemperature: 0.2,
		OllamaNumPredict:  512,
		RequestTimeout:    5 * time.Second,
	})

	summary, err := summarizer.GenerateSummary(
		[]models.Segment{
			{Role: "client", Text: "Заказ должны были привезти утром, но его до сих пор нет."},
			{Role: "operator", Text: "Уточню информацию у службы доставки."},
		},
		"delivery_delay",
		"high",
		&models.Entities{},
	)
	if err != nil {
		t.Fatalf("GenerateSummary returned error: %v", err)
	}

	if summary.Title != "Задержка доставки заказа" {
		t.Fatalf("unexpected title: %q", summary.Title)
	}
	if len(summary.KeyPoints) != 2 {
		t.Fatalf("unexpected key points: %#v", summary.KeyPoints)
	}
	if summary.UrgencyReason == "" {
		t.Fatal("expected urgency reason")
	}
}

func TestGenerateSummaryFallback(t *testing.T) {
	t.Parallel()

	summarizer := NewLLMSummarizer(SummarizerConfig{
		Provider:       "fallback",
		RequestTimeout: 5 * time.Second,
	})

	summary, err := summarizer.GenerateSummary(
		[]models.Segment{
			{Role: "client", Text: "Не могу войти в личный кабинет после смены пароля."},
		},
		"auth_issue",
		"critical",
		nil,
	)
	if err != nil {
		t.Fatalf("GenerateSummary returned error: %v", err)
	}

	if summary.Title == "" {
		t.Fatal("expected non-empty title")
	}
	if !strings.Contains(summary.Description, "Суть обращения") {
		t.Fatalf("unexpected description: %q", summary.Description)
	}
	if summary.UrgencyReason == "" {
		t.Fatal("expected urgency reason for critical priority")
	}
}
