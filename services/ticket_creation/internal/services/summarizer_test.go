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
			Response: `{"problem":"Заказ не доставлен в обещанный срок."}`,
		}
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			t.Fatalf("encode response: %v", err)
		}
	}))
	defer server.Close()

	summarizer := NewLLMSummarizer(SummarizerConfig{
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

	if !strings.Contains(summary.Description, "Проблема: Заказ не доставлен") {
		t.Fatalf("unexpected description: %q", summary.Description)
	}
}

func TestGenerateSummaryRejectsExternalOllamaURL(t *testing.T) {
	t.Parallel()

	summarizer := NewLLMSummarizer(SummarizerConfig{
		OllamaBaseURL:  "https://example.com",
		RequestTimeout: 5 * time.Second,
	})

	_, err := summarizer.GenerateSummary(
		[]models.Segment{
			{Role: "client", Text: "Не могу войти в личный кабинет после смены пароля."},
		},
		"auth_issue",
		"critical",
		nil,
	)
	if err == nil {
		t.Fatal("expected error for external ollama base URL")
	}
}
