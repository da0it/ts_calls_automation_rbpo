package services

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"ticket_module/internal/models"
)

const (
	anthropicAPIURL       = "https://api.anthropic.com/v1/messages"
	anthropicAPIVersion   = "2023-06-01"
	defaultAnthropicModel = "claude-sonnet-4-5-20250929"
	defaultOllamaBaseURL  = "http://localhost:11434"
	defaultOllamaModel    = "qwen2.5:7b"
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
	Provider          string
	AnthropicAPIKey   string
	AnthropicModel    string
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

func NewLLMSummarizer(cfg SummarizerConfig) *LLMSummarizer {
	if strings.TrimSpace(cfg.Provider) == "" {
		cfg.Provider = "ollama"
	}
	if strings.TrimSpace(cfg.AnthropicModel) == "" {
		cfg.AnthropicModel = defaultAnthropicModel
	}
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

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	System    string             `json:"system"`
	Messages  []anthropicMessage `json:"messages"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
}

type anthropicErrorResponse struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
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
	Context string `json:"context"`
	Action  string `json:"action"`
}

func (s *LLMSummarizer) GenerateSummary(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
	switch strings.ToLower(strings.TrimSpace(s.config.Provider)) {
	case "", "ollama":
		return s.generateWithOllama(segments, intentID, priority, entities)
	case "anthropic":
		if strings.TrimSpace(s.config.AnthropicAPIKey) == "" {
			return nil, fmt.Errorf("anthropic provider selected but ANTHROPIC_API_KEY is empty")
		}
		return s.generateWithAnthropic(segments, intentID, priority, entities)
	case "fallback", "none", "disabled":
		return s.fallbackSummary(segments, intentID, priority), nil
	default:
		return nil, fmt.Errorf("unsupported LLM provider: %s", s.config.Provider)
	}
}

func (s *LLMSummarizer) generateWithOllama(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
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

	url := strings.TrimRight(s.config.OllamaBaseURL, "/") + "/api/generate"
	httpReq, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
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

func (s *LLMSummarizer) generateWithAnthropic(
	segments []models.Segment,
	intentID string,
	priority string,
	entities *models.Entities,
) (*models.TicketSummary, error) {
	transcript := formatTranscript(segments)
	entitiesInfo := formatEntities(entities)

	req := anthropicRequest{
		Model:     s.config.AnthropicModel,
		MaxTokens: 1024,
		System:    ticketSummarySystemPrompt(),
		Messages: []anthropicMessage{
			{
				Role:    "user",
				Content: buildSummaryUserPrompt(transcript, intentID, priority, entitiesInfo),
			},
		},
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal anthropic request: %w", err)
	}

	httpReq, err := http.NewRequest(http.MethodPost, anthropicAPIURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create anthropic request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-api-key", s.config.AnthropicAPIKey)
	httpReq.Header.Set("anthropic-version", anthropicAPIVersion)

	resp, err := s.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("anthropic request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read anthropic response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		var errResp anthropicErrorResponse
		if json.Unmarshal(respBody, &errResp) == nil && strings.TrimSpace(errResp.Error.Message) != "" {
			return nil, fmt.Errorf("anthropic error (%d): %s - %s", resp.StatusCode, errResp.Error.Type, errResp.Error.Message)
		}
		return nil, fmt.Errorf("anthropic error (%d): %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	var anthropicResp anthropicResponse
	if err := json.Unmarshal(respBody, &anthropicResp); err != nil {
		return nil, fmt.Errorf("decode anthropic response: %w", err)
	}
	if len(anthropicResp.Content) == 0 || strings.TrimSpace(anthropicResp.Content[0].Text) == "" {
		return nil, fmt.Errorf("empty response from anthropic")
	}

	return parseSummaryJSON(anthropicResp.Content[0].Text, intentID, priority)
}

func parseSummaryJSON(text, intentID, priority string) (*models.TicketSummary, error) {
	jsonStr := strings.TrimSpace(text)
	if start := strings.Index(jsonStr, "{"); start >= 0 {
		if end := strings.LastIndex(jsonStr, "}"); end > start {
			jsonStr = jsonStr[start : end+1]
		}
	}

	var payload structuredSummaryPayload
	if err := json.Unmarshal([]byte(jsonStr), &payload); err == nil {
		if payload.Problem != "" || payload.Context != "" || payload.Action != "" {
			return normalizeSummary(structuredPayloadToSummary(payload), intentID, priority), nil
		}
	}

	var summary models.TicketSummary
	if err := json.Unmarshal([]byte(jsonStr), &summary); err != nil {
		return nil, fmt.Errorf("unmarshal summary JSON: %w", err)
	}

	return normalizeSummary(&summary, intentID, priority), nil
}

func (s *LLMSummarizer) fallbackSummary(
	segments []models.Segment,
	intentID string,
	priority string,
) *models.TicketSummary {
	var clientTexts []string
	for _, seg := range segments {
		if seg.Role == "client" {
			clientTexts = append(clientTexts, seg.Text)
		}
	}

	title := fmt.Sprintf("Обращение: %s", intentID)
	if len(clientTexts) > 0 {
		title = collapseWhitespace(clientTexts[0])
	}

	descriptionParts := []string{"Автоматически созданный тикет из транскрипции звонка."}
	if len(clientTexts) > 0 {
		descriptionParts = append(descriptionParts, fmt.Sprintf("Суть обращения: %s", strings.TrimSpace(clientTexts[0])))
	}

	summary := &models.TicketSummary{
		Title:       title,
		Description: strings.Join(descriptionParts, "\n\n"),
	}
	if priority == "high" || priority == "critical" {
		summary.UrgencyReason = fmt.Sprintf("Приоритет обращения определён как %s.", priority)
	}

	return normalizeSummary(summary, intentID, priority)
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
		description = fmt.Sprintf("Автоматически созданный тикет по обращению %s.", collapseWhitespace(intentID))
	}
	summary.Description = description

	cleanKeyPoints := make([]string, 0, len(summary.KeyPoints))
	for _, point := range summary.KeyPoints {
		if cleaned := collapseWhitespace(point); cleaned != "" {
			cleanKeyPoints = append(cleanKeyPoints, cleaned)
		}
	}
	summary.KeyPoints = cleanKeyPoints
	summary.SuggestedSolution = strings.TrimSpace(summary.SuggestedSolution)
	summary.UrgencyReason = strings.TrimSpace(summary.UrgencyReason)
	if (priority == "high" || priority == "critical") && summary.UrgencyReason == "" {
		summary.UrgencyReason = fmt.Sprintf("Обращение автоматически отмечено как %s.", priority)
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
	lines := make([]string, 0, 3)
	if value := collapseWhitespace(payload.Problem); value != "" {
		lines = append(lines, "Проблема: "+value)
	}
	if value := collapseWhitespace(payload.Context); value != "" {
		lines = append(lines, "Контекст: "+value)
	}
	if value := collapseWhitespace(payload.Action); value != "" {
		lines = append(lines, "Действие: "+value)
	}

	return &models.TicketSummary{
		Description: strings.Join(lines, "\n"),
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
