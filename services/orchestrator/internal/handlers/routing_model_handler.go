package handlers

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

func (h *ProcessHandler) GetRoutingModelStatus(c *gin.Context) {
	if h.routingModelService == nil {
		h.writeAudit(c, "routing.model.status", "routing_model", "", "failed", map[string]interface{}{
			"reason": "service_not_configured",
		})
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "routing model service is not configured"})
		return
	}

	status, err := h.routingModelService.GetStatus()
	if err != nil {
		h.writeAudit(c, "routing.model.status", "routing_model", "", "failed", map[string]interface{}{
			"reason": err.Error(),
		})
		c.JSON(http.StatusBadGateway, gin.H{"error": err.Error()})
		return
	}

	h.writeAudit(c, "routing.model.status", "routing_model", "", "success", map[string]interface{}{})
	c.JSON(http.StatusOK, status)
}
