package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"ticket_module/internal/models"
)

const (
	defaultOllamaBaseURL  = "http://localhost:11434"
	defaultOllamaModel    = "gemma"
	defaultRequestTimeout = 60 * time.Second
	maxTicketTitleRunes   = 100
)

type TicketSummarizer interface {
	GenerateSummary(
		segments []models.Segment,
		intentID string,
		priority string,
		entities *models.Entities,
	) (*models.TicketSummary, error)
}

type SummarizerConfig struct {
	OllamaBaseURL     string
	OllamaModel       string
	OllamaTemperature float64
	OllamaNumPredict  int
	RequestTimeout    time.Duration
}

type LLMSummarizer struct {
	config     SummarizerConfig
	httpClient *http.Client
}

type ollamaRequest struct {
	Model   string         `json:"model"`
	Prompt  string         `json:"prompt"`
	System  string         `json:"system,omitempty"`
	Stream  bool           `json:"stream"`
	Format  string         `json:"format,omitempty"`
	Options *ollamaOptions `json:"options,omitempty"`
}

type ollamaOptions struct {
	Temperature float64 `json:"temperature"`
	NumPredict  int     `json:"num_predict,omitempty"`
}

type ollamaResponse struct {
	Response string `json:"response"`
	Error    string `json:"error"`
}

type structuredSummaryPayload struct {
	Problem string `json:"problem"`
}

func NewLLMSummarizer(cfg SummarizerConfig) *LLMSummarizer {
	if strings.TrimSpace(cfg.OllamaBaseURL) == "" {
		cfg.OllamaBaseURL = defaultOllamaBaseURL
	}
	if strings.TrimSpace(cfg.OllamaModel) == "" {
		cfg.OllamaModel = defaultOllamaModel
	}
	if cfg.RequestTimeout <= 0 {
		cfg.RequestTimeout = defaultRequestTimeout
	}

	return &LLMSummarizer{
		config: cfg,
		httpClient: &http.Client{
			Timeout: cfg.RequestTimeout,
		},
	}
}

func (s *LLMSummarizer) GenerateSummary(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
	return s.generateWithOllama(segments, intentID, priority, entities)
}

func (s *LLMSummarizer) generateWithOllama(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
	if err := validateLocalOllamaBaseURL(s.config.OllamaBaseURL); err != nil {
		return nil, err
	}

	transcript := formatTranscript(segments)
	entitiesInfo := formatEntities(entities)

	req := ollamaRequest{
		Model:  s.config.OllamaModel,
		Prompt: buildSummaryUserPrompt(transcript, intentID, priority, entitiesInfo),
		System: ticketSummarySystemPrompt(),
		Stream: false,
		Format: "json",
		Options: &ollamaOptions{
			Temperature: s.config.OllamaTemperature,
		},
	}
	if s.config.OllamaNumPredict > 0 {
		req.Options.NumPredict = s.config.OllamaNumPredict
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal ollama request: %w", err)
	}

	endpoint := strings.TrimRight(s.config.OllamaBaseURL, "/") + "/api/generate"
	httpReq, err := http.NewRequest(http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create ollama request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read ollama response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama error (%d): %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var ollamaResp ollamaResponse
	if err := json.Unmarshal(respBody, &ollamaResp); err != nil {
		return nil, fmt.Errorf("decode ollama response: %w", err)
	}
	if strings.TrimSpace(ollamaResp.Error) != "" {
		return nil, fmt.Errorf("ollama error: %s", ollamaResp.Error)
	}
	if strings.TrimSpace(ollamaResp.Response) == "" {
		return nil, fmt.Errorf("empty response from ollama")
	}

	return parseSummaryJSON(ollamaResp.Response, intentID, priority)
}

func parseSummaryJSON(text, intentID, priority string) (*models.TicketSummary, error) {
	jsonStr := strings.TrimSpace(text)
	if start := strings.Index(jsonStr, "{"); start >= 0 {
		if end := strings.LastIndex(jsonStr, "}"); end > start {
			jsonStr = jsonStr[start : end+1]
		}
	}

	var payload structuredSummaryPayload
	if err := json.Unmarshal([]byte(jsonStr), &payload); err != nil {
		return nil, fmt.Errorf("unmarshal summary JSON: %w", err)
	}
	if collapseWhitespace(payload.Problem) == "" {
		return nil, fmt.Errorf("empty problem in summary")
	}

	return normalizeSummary(structuredPayloadToSummary(payload), intentID, priority), nil
}

func normalizeSummary(summary *models.TicketSummary, intentID, priority string) *models.TicketSummary {
	if summary == nil {
		summary = &models.TicketSummary{}
	}

	title := collapseWhitespace(summary.Title)
	if title == "" {
		title = fmt.Sprintf("Обращение: %s", collapseWhitespace(intentID))
	}
	summary.Title = truncateRunes(title, maxTicketTitleRunes)

	description := strings.TrimSpace(summary.Description)
	if description == "" {
		description = fmt.Sprintf("Проблема: Автоматически созданный тикет по обращению %s.", collapseWhitespace(intentID))
	}
	summary.Description = description
	summary.KeyPoints = nil
	summary.SuggestedSolution = ""
	summary.UrgencyReason = ""
	if priority == "" {
		return summary
	}
	return summary
}

func truncateRunes(value string, limit int) string {
	runes := []rune(value)
	if len(runes) <= limit {
		return value
	}
	if limit <= 3 {
		return string(runes[:limit])
	}
	return string(runes[:limit-3]) + "..."
}

func collapseWhitespace(value string) string {
	return strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
}

func structuredPayloadToSummary(payload structuredSummaryPayload) *models.TicketSummary {
	return &models.TicketSummary{
		Description: "Проблема: " + collapseWhitespace(payload.Problem),
	}
}

func validateLocalOllamaBaseURL(raw string) error {
	parsed, err := url.Parse(strings.TrimSpace(raw))
	if err != nil {
		return fmt.Errorf("invalid OLLAMA_BASE_URL: %w", err)
	}
	host := strings.ToLower(parsed.Hostname())
	switch host {
	case "localhost", "127.0.0.1", "::1", "ollama", "host.docker.internal":
		return nil
	default:
		return fmt.Errorf("OLLAMA_BASE_URL must point to a local endpoint, got %q", raw)
	}
}

func formatTranscript(segments []models.Segment) string {
	var sb strings.Builder
	for _, seg := range segments {
		role := seg.Role
		if role == "" {
			role = seg.Speaker
		}
		sb.WriteString(fmt.Sprintf("[%s]: %s\n", role, seg.Text))
	}
	return sb.String()
}

func formatEntities(entities *models.Entities) string {
	if entities == nil {
		return "Не извлечены"
	}

	var parts []string
	for _, p := range entities.Persons {
		parts = append(parts, fmt.Sprintf("Имя: %s", p.Value))
	}
	for _, p := range entities.Phones {
		parts = append(parts, fmt.Sprintf("Телефон: %s", p.Value))
	}
	for _, e := range entities.Emails {
		parts = append(parts, fmt.Sprintf("Email: %s", e.Value))
	}
	for _, o := range entities.OrderIDs {
		parts = append(parts, fmt.Sprintf("Заказ: %s", o.Value))
	}
	for _, a := range entities.AccountIDs {
		parts = append(parts, fmt.Sprintf("Аккаунт: %s", a.Value))
	}
	for _, m := range entities.MoneyAmounts {
		parts = append(parts, fmt.Sprintf("Сумма: %s", m.Value))
	}
	for _, d := range entities.Dates {
		parts = append(parts, fmt.Sprintf("Дата: %s", d.Value))
	}

	if len(parts) == 0 {
		return "Не извлечены"
	}

	return strings.Join(parts, "\n")
}
