// internal/services/orchestrator_service.go
package services

import (
	"fmt"
	"log"
	"strings"
	"time"

	"orchestrator/internal/clients"
)

const (
	ProcessStatusCompleted          = "completed"
	ProcessStatusAwaitingSpamReview = "awaiting_spam_review"
	ProcessStatusSpamBlocked        = "spam_blocked"
)

type OrchestratorService struct {
	transcriptionClient *clients.TranscriptionClient
	routingClient       *clients.RoutingClient
	ticketClient        *clients.TicketClient
	notificationClient  *clients.NotificationClient
	entityClient        *clients.EntityClient
}

func NewOrchestratorService(
	transcriptionClient *clients.TranscriptionClient,
	routingClient *clients.RoutingClient,
	ticketClient *clients.TicketClient,
	notificationClient *clients.NotificationClient,
	entityClient *clients.EntityClient,
) *OrchestratorService {
	return &OrchestratorService{
		transcriptionClient: transcriptionClient,
		routingClient:       routingClient,
		ticketClient:        ticketClient,
		notificationClient:  notificationClient,
		entityClient:        entityClient,
	}
}

type ProcessCallResult struct {
	CallID         string                         `json:"call_id"`
	Status         string                         `json:"status"`
	Transcript     *clients.TranscriptionResponse `json:"transcript"`
	Routing        *clients.RoutingResponse       `json:"routing"`
	SpamCheck      *clients.SpamCheckResponse     `json:"spam_check,omitempty"`
	Entities       *clients.Entities              `json:"entities"`
	Ticket         *clients.TicketCreated         `json:"ticket"`
	Notification   *clients.NotificationResult    `json:"notification,omitempty"`
	ProcessingTime map[string]float64             `json:"processing_time"`
	TotalTime      float64                        `json:"total_time"`
}

type ContinueAfterSpamReviewInput struct {
	CallID         string
	SourceFilename string
	Decision       string
	Transcript     *clients.TranscriptionResponse
}

func emptyEntities() *clients.Entities {
	return &clients.Entities{
		Persons:      []clients.ExtractedEntity{},
		Phones:       []clients.ExtractedEntity{},
		Emails:       []clients.ExtractedEntity{},
		OrderIDs:     []clients.ExtractedEntity{},
		AccountIDs:   []clients.ExtractedEntity{},
		MoneyAmounts: []clients.ExtractedEntity{},
		Dates:        []clients.ExtractedEntity{},
	}
}

func normalizeEntities(e *clients.Entities) *clients.Entities {
	if e == nil {
		return emptyEntities()
	}
	if e.Persons == nil {
		e.Persons = []clients.ExtractedEntity{}
	}
	if e.Phones == nil {
		e.Phones = []clients.ExtractedEntity{}
	}
	if e.Emails == nil {
		e.Emails = []clients.ExtractedEntity{}
	}
	if e.OrderIDs == nil {
		e.OrderIDs = []clients.ExtractedEntity{}
	}
	if e.AccountIDs == nil {
		e.AccountIDs = []clients.ExtractedEntity{}
	}
	if e.MoneyAmounts == nil {
		e.MoneyAmounts = []clients.ExtractedEntity{}
	}
	if e.Dates == nil {
		e.Dates = []clients.ExtractedEntity{}
	}
	return e
}

func isSpamReviewRequired(spamCheck *clients.SpamCheckResponse) bool {
	return spamCheck != nil && spamCheck.Status == "review"
}

func isSpamBlocked(spamCheck *clients.SpamCheckResponse) bool {
	return spamCheck != nil && spamCheck.Status == "block"
}

func cloneSpamCheck(spamCheck *clients.SpamCheckResponse) *clients.SpamCheckResponse {
	if spamCheck == nil {
		return nil
	}
	copied := *spamCheck
	return &copied
}

func buildManualSpamCheck(base *clients.SpamCheckResponse, decision string) *clients.SpamCheckResponse {
	out := cloneSpamCheck(base)
	if out == nil {
		out = &clients.SpamCheckResponse{}
	}
	if decision == "spam" {
		out.Status = "block"
		out.PredictedLabel = "spam"
		out.Confidence = 1.0
		out.Reason = "manual_review_marked_spam"
	} else {
		out.Status = "allow"
		out.Reason = "manual_review_marked_not_spam"
	}
	out.Skipped = decision == "not_spam"
	return out
}

func (s *OrchestratorService) routeTranscript(transcript *clients.TranscriptionResponse, skipSpamGate bool) (*clients.RoutingResponse, error) {
	if transcript == nil {
		return nil, fmt.Errorf("transcript is required")
	}
	return s.routingClient.Route(transcript.CallID, transcript.Segments, skipSpamGate)
}

func (s *OrchestratorService) completeNonSpamCall(
	transcript *clients.TranscriptionResponse,
	routing *clients.RoutingResponse,
	status string,
	startTime time.Time,
	processingTime map[string]float64,
) (*ProcessCallResult, error) {
	log.Println("Step 3/5: Extracting entities...")
	stepStart := time.Now()
	entities := emptyEntities()
	if s.entityClient != nil {
		extracted, extractErr := s.entityClient.Extract(transcript.Segments)
		if extractErr != nil {
			log.Printf("⚠ Entity extraction failed (non-fatal): %v", extractErr)
		} else {
			entities = normalizeEntities(extracted)
		}
	}
	processingTime["entity_extraction"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Entity extraction completed in %.2fs (order_ids: %d, phones: %d, emails: %d)",
		processingTime["entity_extraction"], len(entities.OrderIDs), len(entities.Phones), len(entities.Emails))

	log.Println("Step 4/5: Creating ticket...")
	stepStart = time.Now()
	ticket, err := s.ticketClient.CreateTicket(transcript, routing, entities)
	if err != nil {
		return nil, fmt.Errorf("ticket creation failed: %w", err)
	}
	processingTime["ticket_creation"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Ticket created in %.2fs (ID: %s, URL: %s)",
		processingTime["ticket_creation"], ticket.TicketID, ticket.URL)

	log.Println("Step 5/5: Sending notifications...")
	stepStart = time.Now()
	var notification *clients.NotificationResult
	notification, err = s.notificationClient.SendNotification(transcript, routing, entities, ticket)
	if err != nil {
		log.Printf("⚠ Notification sending failed (non-fatal): %v", err)
		notification = &clients.NotificationResult{Success: false}
	}
	processingTime["notification"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Notifications sent in %.2fs (success: %v)",
		processingTime["notification"], notification.Success)

	totalTime := time.Since(startTime).Seconds()
	log.Printf("Call processing completed successfully in %.2fs", totalTime)

	return &ProcessCallResult{
		CallID:         transcript.CallID,
		Status:         status,
		Transcript:     transcript,
		Routing:        routing,
		SpamCheck:      cloneSpamCheck(routing.SpamCheck),
		Entities:       entities,
		Ticket:         ticket,
		Notification:   notification,
		ProcessingTime: processingTime,
		TotalTime:      totalTime,
	}, nil
}

// ProcessCall обрабатывает аудио звонка через все модули
func (s *OrchestratorService) ProcessCall(audioPath string) (*ProcessCallResult, error) {
	startTime := time.Now()
	processingTime := make(map[string]float64)

	log.Printf("Starting call processing for audio: %s", audioPath)

	// 1. Транскрибация
	log.Println("Step 1/5: Transcribing audio...")
	stepStart := time.Now()
	transcript, err := s.transcriptionClient.Transcribe(audioPath)
	if err != nil {
		return nil, fmt.Errorf("transcription failed: %w", err)
	}
	processingTime["transcription"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Transcription completed in %.2fs (found %d segments)",
		processingTime["transcription"], len(transcript.Segments))

	// 2. Маршрутизация
	log.Println("Step 2/5: Routing call...")
	stepStart = time.Now()
	routing, err := s.routeTranscript(transcript, false)
	if err != nil {
		return nil, fmt.Errorf("routing failed: %w", err)
	}
	processingTime["routing"] = time.Since(stepStart).Seconds()
	log.Printf("✓ Routing completed in %.2fs (intent: %s, priority: %s)",
		processingTime["routing"], routing.IntentID, routing.Priority)

	if isSpamReviewRequired(routing.SpamCheck) {
		totalTime := time.Since(startTime).Seconds()
		log.Printf("Call processing paused for manual spam review in %.2fs", totalTime)
		return &ProcessCallResult{
			CallID:         transcript.CallID,
			Status:         ProcessStatusAwaitingSpamReview,
			Transcript:     transcript,
			Routing:        routing,
			SpamCheck:      cloneSpamCheck(routing.SpamCheck),
			Entities:       emptyEntities(),
			ProcessingTime: processingTime,
			TotalTime:      totalTime,
		}, nil
	}
	if isSpamBlocked(routing.SpamCheck) {
		totalTime := time.Since(startTime).Seconds()
		log.Printf("Call blocked by spam gate in %.2fs", totalTime)
		return &ProcessCallResult{
			CallID:         transcript.CallID,
			Status:         ProcessStatusSpamBlocked,
			Transcript:     transcript,
			Routing:        routing,
			SpamCheck:      cloneSpamCheck(routing.SpamCheck),
			Entities:       emptyEntities(),
			ProcessingTime: processingTime,
			TotalTime:      totalTime,
		}, nil
	}
	return s.completeNonSpamCall(transcript, routing, ProcessStatusCompleted, startTime, processingTime)
}

func (s *OrchestratorService) ContinueAfterSpamReview(input ContinueAfterSpamReviewInput) (*ProcessCallResult, error) {
	startTime := time.Now()
	processingTime := make(map[string]float64)
	decision := strings.ToLower(strings.TrimSpace(input.Decision))

	if input.Transcript == nil {
		return nil, fmt.Errorf("transcript is required")
	}
	if input.Transcript.CallID == "" {
		input.Transcript.CallID = input.CallID
	}
	if input.Transcript.CallID == "" {
		input.Transcript.CallID = "unknown-call"
	}

	switch decision {
	case "spam":
		spamCheck := buildManualSpamCheck(nil, "spam")
		totalTime := time.Since(startTime).Seconds()
		return &ProcessCallResult{
			CallID:     input.Transcript.CallID,
			Status:     ProcessStatusSpamBlocked,
			Transcript: input.Transcript,
			Routing: &clients.RoutingResponse{
				IntentID:         "spam",
				IntentConfidence: 1.0,
				Priority:         "high",
				SpamCheck:        spamCheck,
			},
			SpamCheck:      spamCheck,
			Entities:       emptyEntities(),
			ProcessingTime: processingTime,
			TotalTime:      totalTime,
		}, nil
	case "not_spam":
		log.Println("Step 2/5: Routing call after manual spam review...")
		stepStart := time.Now()
		routing, err := s.routeTranscript(input.Transcript, true)
		if err != nil {
			return nil, fmt.Errorf("routing after spam review failed: %w", err)
		}
		if routing.SpamCheck == nil {
			routing.SpamCheck = buildManualSpamCheck(nil, "not_spam")
		} else {
			routing.SpamCheck = buildManualSpamCheck(routing.SpamCheck, "not_spam")
		}
		processingTime["routing"] = time.Since(stepStart).Seconds()
		log.Printf("✓ Routing after spam review completed in %.2fs (intent: %s, priority: %s)",
			processingTime["routing"], routing.IntentID, routing.Priority)

		return s.completeNonSpamCall(input.Transcript, routing, ProcessStatusCompleted, startTime, processingTime)
	default:
		return nil, fmt.Errorf("decision must be spam or not_spam")
	}
}

// HealthCheck проверяет доступность всех сервисов
func (s *OrchestratorService) HealthCheck() map[string]string {
	// TODO: можно добавить проверку health endpoints всех сервисов
	return map[string]string{
		"orchestrator":  "healthy",
		"transcription": "unknown",
		"routing":       "unknown",
		"ticket":        "unknown",
	}
}
