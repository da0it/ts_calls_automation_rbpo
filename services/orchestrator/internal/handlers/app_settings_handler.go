package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

type updateAppSettingsRequest struct {
	SLAMinutes int `json:"sla_minutes"`
}

func (h *ProcessHandler) GetAppSettings(c *gin.Context) {
	if h.appSettingsService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "app settings service is not configured"})
		return
	}

	settings, err := h.appSettingsService.Get()
	if err != nil {
		h.writeAudit(c, "app.settings.get", "app_settings", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "app.settings.get", "app_settings", "", "success", map[string]interface{}{
		"sla_minutes": settings.SLAMinutes,
	})
	c.JSON(http.StatusOK, settings)
}

func (h *ProcessHandler) UpdateAppSettings(c *gin.Context) {
	if h.appSettingsService == nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "app settings service is not configured"})
		return
	}

	var payload updateAppSettingsRequest
	if err := c.ShouldBindJSON(&payload); err != nil {
		h.writeAudit(c, "app.settings.update", "app_settings", "", "failed", map[string]interface{}{
			"reason": "invalid_payload",
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body"})
		return
	}

	settings, err := h.appSettingsService.UpdateSLA(payload.SLAMinutes)
	if err != nil {
		h.writeAudit(c, "app.settings.update", "app_settings", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "app.settings.update", "app_settings", "", "success", map[string]interface{}{
		"sla_minutes": settings.SLAMinutes,
	})
	c.JSON(http.StatusOK, settings)
}
