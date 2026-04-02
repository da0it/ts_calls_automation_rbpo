package adapters

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"ticket_module/internal/models"
)

const defaultWebhookTimeout = 30 * time.Second

type WebhookAdapterConfig struct {
	EndpointURL    string
	Headers        map[string]string
	Timeout        time.Duration
	SystemName     string
	ExternalIDPath string
	URLPath        string
}

type WebhookAdapter struct {
	endpointURL    string
	headers        map[string]string
	httpClient     *http.Client
	systemName     string
	externalIDPath string
	urlPath        string
}

func NewWebhookAdapter(cfg WebhookAdapterConfig) (*WebhookAdapter, error) {
	endpointURL := strings.TrimSpace(cfg.EndpointURL)
	if endpointURL == "" {
		return nil, fmt.Errorf("webhook endpoint URL is required")
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = defaultWebhookTimeout
	}

	return &WebhookAdapter{
		endpointURL:    endpointURL,
		headers:        cloneHeaders(cfg.Headers),
		httpClient:     &http.Client{Timeout: cfg.Timeout},
		systemName:     normalizeSystemName(cfg.SystemName),
		externalIDPath: strings.TrimSpace(cfg.ExternalIDPath),
		urlPath:        strings.TrimSpace(cfg.URLPath),
	}, nil
}

func ParseHeadersJSON(raw string) (map[string]string, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return map[string]string{}, nil
	}

	var payload map[string]interface{}
	if err := json.Unmarshal([]byte(raw), &payload); err != nil {
		return nil, fmt.Errorf("parse headers JSON: %w", err)
	}

	headers := make(map[string]string, len(payload))
	for key, value := range payload {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		text := stringifyPathValue(value)
		if text == "" {
			return nil, fmt.Errorf("header %q must be a scalar string/number/bool", key)
		}
		headers[key] = text
	}

	return headers, nil
}

func (a *WebhookAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	if payload == nil {
		return nil, fmt.Errorf("ticket payload is required")
	}
	if payload.Draft == nil {
		return nil, fmt.Errorf("ticket payload draft is required")
	}

	requestBody, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal webhook payload: %w", err)
	}

	req, err := http.NewRequest(http.MethodPost, a.endpointURL, bytes.NewReader(requestBody))
	if err != nil {
		return nil, fmt.Errorf("create webhook request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	req.Header.Set("User-Agent", "ts-calls-automation/ticket-service")
	for key, value := range a.headers {
		req.Header.Set(key, value)
	}

	resp, err := a.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send webhook request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read webhook response: %w", err)
	}
	if resp.StatusCode < http.StatusOK || resp.StatusCode >= http.StatusMultipleChoices {
		return nil, fmt.Errorf("%s response %d: %s", a.systemName, resp.StatusCode, compactResponseBody(respBody))
	}

	return a.buildCreated(respBody, payload), nil
}

func (a *WebhookAdapter) GetTicket(externalID string) (*models.TicketCreated, error) {
	return nil, fmt.Errorf("%s get ticket is not implemented", a.systemName)
}

func (a *WebhookAdapter) UpdateTicket(externalID string, update map[string]interface{}) error {
	return fmt.Errorf("%s update ticket is not implemented", a.systemName)
}

func (a *WebhookAdapter) buildCreated(respBody []byte, payload *models.TicketSystemPayload) *models.TicketCreated {
	created := &models.TicketCreated{
		ExternalID: a.fallbackExternalID(payload),
		System:     a.systemName,
		CreatedAt:  time.Now().UTC(),
	}

	if strings.TrimSpace(string(respBody)) == "" {
		return created
	}

	var decoded interface{}
	if err := json.Unmarshal(respBody, &decoded); err != nil {
		return created
	}

	if externalID := firstNonEmpty(
		lookupPathString(decoded, a.externalIDPath),
		lookupResponseString(decoded, "external_id", "ticket_id", "number", "sys_id", "record_id", "id"),
	); externalID != "" {
		created.ExternalID = externalID
	}
	if recordURL := firstNonEmpty(
		lookupPathString(decoded, a.urlPath),
		lookupResponseString(decoded, "url", "record_url", "recordUrl", "href"),
	); recordURL != "" {
		created.URL = recordURL
	}

	return created
}

func (a *WebhookAdapter) fallbackExternalID(payload *models.TicketSystemPayload) string {
	prefix := normalizeSystemName(a.systemName)
	if payload != nil && payload.Service.CallID != "" {
		return truncateIdentifier(prefix+"-ack-"+payload.Service.CallID, 128)
	}
	return truncateIdentifier(prefix+"-ack-"+uuid.NewString(), 128)
}

func cloneHeaders(src map[string]string) map[string]string {
	if len(src) == 0 {
		return map[string]string{}
	}

	dst := make(map[string]string, len(src))
	for key, value := range src {
		key = strings.TrimSpace(key)
		value = strings.TrimSpace(value)
		if key == "" || value == "" {
			continue
		}
		dst[key] = value
	}
	return dst
}

func normalizeSystemName(value string) string {
	value = strings.TrimSpace(strings.ToLower(value))
	if value == "" {
		return "webhook"
	}

	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_':
			b.WriteRune(r)
		case r == ' ':
			b.WriteRune('-')
		}
	}

	if b.Len() == 0 {
		return "webhook"
	}
	return b.String()
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if strings.TrimSpace(value) != "" {
			return strings.TrimSpace(value)
		}
	}
	return ""
}

func truncateIdentifier(value string, limit int) string {
	if len(value) <= limit {
		return value
	}
	if limit <= 0 {
		return ""
	}
	return value[:limit]
}

func lookupPathString(value interface{}, path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}

	current := value
	for _, part := range strings.Split(path, ".") {
		part = strings.TrimSpace(part)
		if part == "" {
			return ""
		}

		switch typed := current.(type) {
		case map[string]interface{}:
			next, ok := typed[part]
			if !ok {
				return ""
			}
			current = next
		case []interface{}:
			index, err := strconv.Atoi(part)
			if err != nil || index < 0 || index >= len(typed) {
				return ""
			}
			current = typed[index]
		default:
			return ""
		}
	}

	return stringifyPathValue(current)
}

func lookupResponseString(value interface{}, keys ...string) string {
	for _, candidate := range collectResponseMaps(value) {
		for _, key := range keys {
			if raw, ok := candidate[key]; ok {
				if text := stringifyPathValue(raw); text != "" {
					return text
				}
			}
		}
	}
	return ""
}

func collectResponseMaps(value interface{}) []map[string]interface{} {
	var out []map[string]interface{}

	switch typed := value.(type) {
	case map[string]interface{}:
		out = append(out, typed)
		for _, nested := range typed {
			out = append(out, collectResponseMaps(nested)...)
		}
	case []interface{}:
		for _, nested := range typed {
			out = append(out, collectResponseMaps(nested)...)
		}
	}

	return out
}

func stringifyPathValue(value interface{}) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case bool:
		return strconv.FormatBool(typed)
	case float64:
		return strings.TrimSpace(strconv.FormatFloat(typed, 'f', -1, 64))
	case json.Number:
		return strings.TrimSpace(typed.String())
	case fmt.Stringer:
		return strings.TrimSpace(typed.String())
	default:
		return ""
	}
}

func compactResponseBody(body []byte) string {
	text := strings.Join(strings.Fields(strings.TrimSpace(string(body))), " ")
	if text == "" {
		return "empty body"
	}
	const maxLen = 300
	if len(text) <= maxLen {
		return text
	}
	return text[:maxLen] + "..."
}
