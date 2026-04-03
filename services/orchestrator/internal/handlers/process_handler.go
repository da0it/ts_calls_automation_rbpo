// internal/handlers/process_handler.go
package handlers

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"orchestrator/internal/clients"
	"orchestrator/internal/models"
	"orchestrator/internal/services"
)

type ProcessHandler struct {
	orchestrator           *services.OrchestratorService
	appSettingsService     *services.AppSettingsService
	routingConfigService   *services.RoutingConfigService
	routingFeedbackService *services.RoutingFeedbackService
	spamFeedbackService    *services.SpamFeedbackService
	routingModelService    *services.RoutingModelService
	auditService           *services.AuditService
	uploadDir              string
}

func envBool(name string, def bool) bool {
	raw := strings.TrimSpace(strings.ToLower(os.Getenv(name)))
	if raw == "" {
		return def
	}
	switch raw {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return def
	}
}

func NewProcessHandler(
	orchestrator *services.OrchestratorService,
	appSettingsService *services.AppSettingsService,
	routingConfigService *services.RoutingConfigService,
	routingFeedbackService *services.RoutingFeedbackService,
	spamFeedbackService *services.SpamFeedbackService,
	routingModelService *services.RoutingModelService,
	auditService *services.AuditService,
) *ProcessHandler {
	// Создаём директорию для загрузки файлов
	uploadDir := "./uploads"
	os.MkdirAll(uploadDir, 0755)

	return &ProcessHandler{
		orchestrator:           orchestrator,
		appSettingsService:     appSettingsService,
		routingConfigService:   routingConfigService,
		routingFeedbackService: routingFeedbackService,
		spamFeedbackService:    spamFeedbackService,
		routingModelService:    routingModelService,
		auditService:           auditService,
		uploadDir:              uploadDir,
	}
}

func (h *ProcessHandler) writeAudit(
	c *gin.Context,
	eventType string,
	resourceType string,
	resourceID string,
	outcome string,
	details map[string]interface{},
) {
	if h.auditService == nil {
		return
	}

	var actorUserID *int64
	actorUsername := ""
	actorRole := ""
	if userVal, ok := c.Get("user"); ok {
		if user, castOK := userVal.(*models.User); castOK && user != nil {
			actorUserID = &user.ID
			actorUsername = user.Username
			actorRole = string(user.Role)
		}
	}

	if err := h.auditService.LogEvent(services.AuditEvent{
		RequestID:     c.GetString("request_id"),
		ActorUserID:   actorUserID,
		ActorUsername: actorUsername,
		ActorRole:     actorRole,
		EventType:     eventType,
		ResourceType:  resourceType,
		ResourceID:    resourceID,
		Outcome:       outcome,
		Details:       details,
		IPAddress:     c.ClientIP(),
		UserAgent:     c.GetHeader("User-Agent"),
	}); err != nil {
		log.Printf("Failed to write audit event (%s): %v", eventType, err)
	}
}

// ProcessCall godoc
// @Summary Обработать аудио звонка
// @Description Загружает аудио файл, транскрибирует, определяет интент и создает тикет
// @Tags calls
// @Accept multipart/form-data
// @Produce json
// @Param audio formData file true "Audio file (mp3, wav, m4a)"
// @Success 200 {object} services.ProcessCallResult
// @Failure 400 {object} map[string]string
// @Failure 500 {object} map[string]string
// @Router /api/v1/process-call [post]
func (h *ProcessHandler) ProcessCall(c *gin.Context) {
	requestReceivedAt := time.Now().UTC()

	// 1. Получаем загруженный файл
	file, err := c.FormFile("audio")
	if err != nil {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason": "missing_audio",
		})
		c.JSON(http.StatusBadRequest, gin.H{
			"error": "audio file is required",
		})
		return
	}

	originalName := filepath.Base(file.Filename)
	ext := strings.ToLower(filepath.Ext(originalName))
	log.Printf(
		"Received audio upload request_id=%s ext=%s size_mb=%.2f",
		c.GetString("request_id"),
		ext,
		float64(file.Size)/1024/1024,
	)

	// Валидация формата
	allowedFormats := map[string]bool{
		".mp3":  true,
		".wav":  true,
		".m4a":  true,
		".flac": true,
		".ogg":  true,
	}
	if !allowedFormats[ext] {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "unsupported_audio_format",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusBadRequest, gin.H{
			"error": fmt.Sprintf("unsupported audio format: %s (allowed: mp3, wav, m4a, flac, ogg)", ext),
		})
		return
	}

	// 2. Сохраняем файл
	requestID := c.GetString("request_id")
	if requestID == "" {
		requestID = "no_request_id"
	}
	filename := fmt.Sprintf("%s_%d%s", requestID, time.Now().UnixNano(), ext)
	audioPath := filepath.Join(h.uploadDir, filename)

	if err := c.SaveUploadedFile(file, audioPath); err != nil {
		log.Printf("Failed to save file: %v", err)
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "save_upload_failed",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "failed to save audio file",
		})
		return
	}

	log.Printf("Saved uploaded audio request_id=%s", requestID)
	deleteAfterProcess := envBool("ORCH_DELETE_UPLOADED_AUDIO_AFTER_PROCESS", true)
	if deleteAfterProcess {
		defer func() {
			if rmErr := os.Remove(audioPath); rmErr != nil && !os.IsNotExist(rmErr) {
				log.Printf("Failed to remove uploaded audio %s: %v", audioPath, rmErr)
			}
		}()
	}

	// 3. Запускаем обработку
	result, err := h.orchestrator.ProcessCall(audioPath)
	if err != nil {
		log.Printf("Processing failed: %v", err)

		// Удаляем файл при ошибке, даже если cleanup on success выключен.
		_ = os.Remove(audioPath)
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason":     "pipeline_failed",
			"audio_ext":  ext,
			"audio_size": file.Size,
		})

		c.JSON(http.StatusInternalServerError, gin.H{
			"error": fmt.Sprintf("processing failed: %v", err),
		})
		return
	}
	if result == nil {
		h.writeAudit(c, "call.process", "call", "", "failed", map[string]interface{}{
			"reason": "empty_pipeline_result",
		})
		c.JSON(http.StatusInternalServerError, gin.H{
			"error": "processing failed: empty pipeline result",
		})
		return
	}

	result.RequestReceivedAt = requestReceivedAt.Format(time.RFC3339Nano)
	result.ProcessedAt = time.Now().UTC().Format(time.RFC3339Nano)

	// 4. Опционально: удаляем файл после обработки
	// os.Remove(audioPath)
	segmentsCount := 0
	intentID := ""
	priority := ""
	suggestedGroup := ""
	if result.Transcript != nil {
		segmentsCount = len(result.Transcript.Segments)
	}
	if result.Routing != nil {
		intentID = result.Routing.IntentID
		priority = result.Routing.Priority
		suggestedGroup = result.Routing.SuggestedGroup
	}
	h.writeAudit(c, "call.process", "call", result.CallID, "success", map[string]interface{}{
		"audio_ext":       ext,
		"audio_size":      file.Size,
		"segments_count":  segmentsCount,
		"status":          result.Status,
		"intent_id":       intentID,
		"priority":        priority,
		"suggested_group": suggestedGroup,
	})

	c.JSON(http.StatusOK, result)
}

type spamReviewTranscriptPayload struct {
	CallID      string                 `json:"call_id"`
	Segments    []clients.Segment      `json:"segments"`
	RoleMapping map[string]string      `json:"role_mapping"`
	Metadata    map[string]interface{} `json:"metadata"`
}

type spamReviewRequest struct {
	CallID         string                      `json:"call_id"`
	SourceFilename string                      `json:"source_filename"`
	Decision       string                      `json:"decision"`
	Transcript     spamReviewTranscriptPayload `json:"transcript"`
	SpamCheck      struct {
		Status         string  `json:"status"`
		PredictedLabel string  `json:"predicted_label"`
		Confidence     float64 `json:"confidence"`
		ThresholdLow   float64 `json:"threshold_low"`
		ThresholdHigh  float64 `json:"threshold_high"`
		Reason         string  `json:"reason"`
		Backend        string  `json:"backend"`
	} `json:"spam_check"`
}

type routingReviewRequest struct {
	CallID         string                      `json:"call_id"`
	SourceFilename string                      `json:"source_filename"`
	Decision       string                      `json:"decision"`
	Transcript     spamReviewTranscriptPayload `json:"transcript"`
	Routing        struct {
		IntentID         string  `json:"intent_id"`
		IntentConfidence float64 `json:"intent_confidence"`
		Priority         string  `json:"priority"`
		SuggestedGroup   string  `json:"suggested_group"`
	} `json:"routing"`
	SpamCheck struct {
		Status         string  `json:"status"`
		PredictedLabel string  `json:"predicted_label"`
		Confidence     float64 `json:"confidence"`
		ThresholdLow   float64 `json:"threshold_low"`
		ThresholdHigh  float64 `json:"threshold_high"`
		Reason         string  `json:"reason"`
		Backend        string  `json:"backend"`
	} `json:"spam_check"`
}

func (h *ProcessHandler) ResolveSpamReview(c *gin.Context) {
	if h.spamFeedbackService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "spam feedback service is not configured"})
		return
	}

	var payload spamReviewRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	transcript := &clients.TranscriptionResponse{
		CallID:      strings.TrimSpace(payload.Transcript.CallID),
		Segments:    payload.Transcript.Segments,
		RoleMapping: payload.Transcript.RoleMapping,
		Metadata:    payload.Transcript.Metadata,
	}
	if transcript.CallID == "" {
		transcript.CallID = strings.TrimSpace(payload.CallID)
	}
	if transcript.Metadata == nil {
		transcript.Metadata = map[string]interface{}{}
	}

	feedbackSegments := make([]services.FeedbackTranscriptSegment, 0, len(transcript.Segments))
	for _, seg := range transcript.Segments {
		feedbackSegments = append(feedbackSegments, services.FeedbackTranscriptSegment{
			Start:   seg.Start,
			End:     seg.End,
			Speaker: seg.Speaker,
			Role:    seg.Role,
			Text:    seg.Text,
		})
	}

	if _, err := h.spamFeedbackService.SaveDecision(services.SpamGateFeedbackRequest{
		CallID:             transcript.CallID,
		SourceFilename:     payload.SourceFilename,
		Decision:           payload.Decision,
		TranscriptText:     segmentsToPlainText(transcript.Segments, 8000),
		TranscriptSegments: feedbackSegments,
		TrainingSample:     segmentsToPlainText(transcript.Segments, 280),
		SpamCheck: services.SpamGateFeedbackMeta{
			Status:         payload.SpamCheck.Status,
			PredictedLabel: payload.SpamCheck.PredictedLabel,
			Confidence:     payload.SpamCheck.Confidence,
			ThresholdLow:   payload.SpamCheck.ThresholdLow,
			ThresholdHigh:  payload.SpamCheck.ThresholdHigh,
			Reason:         payload.SpamCheck.Reason,
			Backend:        payload.SpamCheck.Backend,
		},
	}); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	result, err := h.orchestrator.ContinueAfterSpamReview(services.ContinueAfterSpamReviewInput{
		CallID:         payload.CallID,
		SourceFilename: payload.SourceFilename,
		Decision:       payload.Decision,
		Transcript:     transcript,
	})
	if err != nil {
		h.writeAudit(c, "call.spam_review", "call", transcript.CallID, "failed", map[string]interface{}{
			"decision": payload.Decision,
			"reason":   err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.spam_review", "call", transcript.CallID, "success", map[string]interface{}{
		"decision": payload.Decision,
		"status":   result.Status,
	})

	c.JSON(http.StatusOK, result)
}

func (h *ProcessHandler) ResolveRoutingReview(c *gin.Context) {
	var payload routingReviewRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	transcript := &clients.TranscriptionResponse{
		CallID:      strings.TrimSpace(payload.Transcript.CallID),
		Segments:    payload.Transcript.Segments,
		RoleMapping: payload.Transcript.RoleMapping,
		Metadata:    payload.Transcript.Metadata,
	}
	if transcript.CallID == "" {
		transcript.CallID = strings.TrimSpace(payload.CallID)
	}
	if transcript.Metadata == nil {
		transcript.Metadata = map[string]interface{}{}
	}

	routing := &clients.RoutingResponse{
		IntentID:         strings.TrimSpace(payload.Routing.IntentID),
		IntentConfidence: payload.Routing.IntentConfidence,
		Priority:         strings.TrimSpace(payload.Routing.Priority),
		SuggestedGroup:   strings.TrimSpace(payload.Routing.SuggestedGroup),
	}
	if payload.SpamCheck.Status != "" ||
		payload.SpamCheck.PredictedLabel != "" ||
		payload.SpamCheck.Confidence != 0 ||
		payload.SpamCheck.ThresholdLow != 0 ||
		payload.SpamCheck.ThresholdHigh != 0 ||
		payload.SpamCheck.Reason != "" ||
		payload.SpamCheck.Backend != "" {
		routing.SpamCheck = &clients.SpamCheckResponse{
			Status:         payload.SpamCheck.Status,
			PredictedLabel: payload.SpamCheck.PredictedLabel,
			Confidence:     payload.SpamCheck.Confidence,
			ThresholdLow:   payload.SpamCheck.ThresholdLow,
			ThresholdHigh:  payload.SpamCheck.ThresholdHigh,
			Reason:         payload.SpamCheck.Reason,
			Backend:        payload.SpamCheck.Backend,
		}
	}

	result, err := h.orchestrator.ContinueAfterRoutingReview(services.ContinueAfterRoutingReviewInput{
		CallID:         payload.CallID,
		SourceFilename: payload.SourceFilename,
		Decision:       payload.Decision,
		Transcript:     transcript,
		Routing:        routing,
	})
	if err != nil {
		h.writeAudit(c, "call.routing_review", "call", transcript.CallID, "failed", map[string]interface{}{
			"decision": payload.Decision,
			"reason":   err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "call.routing_review", "call", transcript.CallID, "success", map[string]interface{}{
		"decision":        payload.Decision,
		"status":          result.Status,
		"intent_id":       routing.IntentID,
		"priority":        routing.Priority,
		"suggested_group": routing.SuggestedGroup,
	})

	c.JSON(http.StatusOK, result)
}

func segmentsToPlainText(segments []clients.Segment, maxChars int) string {
	if len(segments) == 0 {
		return ""
	}
	parts := make([]string, 0, len(segments))
	for _, segment := range segments {
		text := strings.TrimSpace(segment.Text)
		if text != "" {
			parts = append(parts, text)
		}
	}
	joined := strings.Join(parts, " ")
	if maxChars > 0 && len(joined) > maxChars {
		return joined[:maxChars]
	}
	return joined
}

// Health godoc
// @Summary Health check
// @Description Проверка доступности оркестратора и зависимых сервисов
// @Tags health
// @Produce json
// @Success 200 {object} map[string]string
// @Router /health [get]
func (h *ProcessHandler) Health(c *gin.Context) {
	status := h.orchestrator.HealthCheck()
	c.JSON(http.StatusOK, status)
}

// Root godoc
// @Summary API Information
// @Description Информация об Orchestrator API
// @Tags info
// @Produce json
// @Success 200 {object} map[string]interface{}
// @Router /api/info [get]
func (h *ProcessHandler) Root(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{
		"service":     "Ticket System Orchestrator",
		"version":     "1.0.0",
		"description": "Оркестрирует обработку звонков через все модули системы",
		"endpoints": gin.H{
			"process_call":     "POST /api/v1/process-call",
			"app_settings":     "GET /api/v1/app-settings, PUT /api/v1/app-settings (admin)",
			"routing_config":   "GET/PUT /api/v1/routing-config",
			"routing_groups":   "POST/DELETE /api/v1/routing-config/groups",
			"routing_intents":  "DELETE /api/v1/routing-config/intents/:id",
			"routing_feedback": "POST /api/v1/routing-feedback",
			"spam_review":      "POST /api/v1/spam-review",
			"routing_review":   "POST /api/v1/routing-review",
			"routing_model":    "GET /api/v1/routing-model/status, POST /api/v1/routing-model/reload, POST /api/v1/routing-model/train, POST /api/v1/routing-model/train-csv",
			"audit_events":     "GET /api/v1/audit/events (admin)",
			"health":           "GET /health",
			"docs":             "GET /docs (если включен Swagger)",
		},
		"pipeline": []string{
			"1. Transcription + Diarization",
			"2. Spam Gate (allow / block / manual review)",
			"3. Routing (RuBERT Intent Classification + low-confidence review)",
			"4. Ticket Creation",
		},
	})
}
