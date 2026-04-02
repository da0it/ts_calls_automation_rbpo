package adapters

import (
	"fmt"
	"time"

	"ticket_module/internal/models"
)

type SimpleOneAdapterConfig struct {
	EndpointURL string
	BearerToken string
	Timeout     time.Duration
}

type SimpleOneAdapter struct {
	webhook *WebhookAdapter
}

func NewSimpleOneAdapter(cfg SimpleOneAdapterConfig) (*SimpleOneAdapter, error) {
	headers := map[string]string{}
	if cfg.BearerToken != "" {
		headers["Authorization"] = "Bearer " + cfg.BearerToken
	}

	webhook, err := NewWebhookAdapter(WebhookAdapterConfig{
		EndpointURL: cfg.EndpointURL,
		Headers:     headers,
		Timeout:     cfg.Timeout,
		SystemName:  "simpleone",
	})
	if err != nil {
		return nil, err
	}

	return &SimpleOneAdapter{webhook: webhook}, nil
}

func (a *SimpleOneAdapter) CreateTicket(payload *models.TicketSystemPayload) (*models.TicketCreated, error) {
	if a == nil || a.webhook == nil {
		return nil, fmt.Errorf("simpleone adapter is not configured")
	}
	return a.webhook.CreateTicket(payload)
}

func (a *SimpleOneAdapter) GetTicket(externalID string) (*models.TicketCreated, error) {
	if a == nil || a.webhook == nil {
		return nil, fmt.Errorf("simpleone adapter is not configured")
	}
	return a.webhook.GetTicket(externalID)
}

func (a *SimpleOneAdapter) UpdateTicket(externalID string, update map[string]interface{}) error {
	if a == nil || a.webhook == nil {
		return fmt.Errorf("simpleone adapter is not configured")
	}
	return a.webhook.UpdateTicket(externalID, update)
}
