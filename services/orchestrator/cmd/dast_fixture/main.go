package main

import (
	"log"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"os/exec"
	"github.com/gin-gonic/gin"
	"orchestrator/internal/handlers"
	"orchestrator/internal/services"
)

const fixtureToken = "test-admin-token"

type authRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

func main() {
	gin.SetMode(gin.ReleaseMode)

	router := gin.New()
	router.Use(gin.Logger(), gin.Recovery())

	processHandler := handlers.NewProcessHandler(
		services.NewOrchestratorService(nil, nil, nil, nil, 0.8),
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
		nil,
	)

	webDir, err := fixtureWebDir()
	if err != nil {
		log.Fatalf("resolve fixture web dir: %v", err)
	}

	router.GET("/", func(c *gin.Context) {
		c.File(filepath.Join(webDir, "index.html"))
	})
	router.GET("/health", processHandler.Health)
	router.GET("/api/info", processHandler.Root)


	router.POST("/api/v1/auth/login", func(c *gin.Context) {
		var req authRequest
		if err := c.ShouldBindJSON(&req); err != nil || strings.TrimSpace(req.Username) == "" || strings.TrimSpace(req.Password) == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
			return
		}
		c.JSON(http.StatusOK, gin.H{
			"token": fixtureToken,
			"user": gin.H{
				"id":          1,
				"username":    req.Username,
				"role":        "admin",
				"is_active":   true,
				"is_approved": true,
			},
		})
	})

	router.POST("/api/v1/auth/register", func(c *gin.Context) {
		var req authRequest
		if err := c.ShouldBindJSON(&req); err != nil || strings.TrimSpace(req.Username) == "" || strings.TrimSpace(req.Password) == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "username and password are required"})
			return
		}
		c.JSON(http.StatusCreated, gin.H{
			"message": "registration submitted, wait for admin approval",
			"user": gin.H{
				"id":          2,
				"username":    req.Username,
				"role":        "operator",
				"is_active":   false,
				"is_approved": false,
			},
		})
	})

	api := router.Group("/api/v1")
	api.Use(mockAuth())
	{
		api.GET("/auth/me", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{
				"id":          1,
				"username":    "admin",
				"role":        "admin",
				"is_active":   true,
				"is_approved": true,
			})
		})
		api.POST("/process-call", processHandler.ProcessCall)
		api.GET("/calls", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"items": []gin.H{}, "total": 0})
		})
		api.POST("/spam-override", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"message": "spam override saved"})
		})
		api.POST("/routing-review", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"message": "routing review saved"})
		})
		api.GET("/app-settings", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"sla_minutes": 15})
		})
		api.GET("/routing-config", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{
				"intents": []gin.H{
					{"id": "orders.problem", "priority": "high"},
					{"id": "misc.triage", "priority": "medium"},
				},
			})
		})
		api.POST("/routing-feedback", func(c *gin.Context) {
			c.JSON(http.StatusCreated, gin.H{"message": "routing feedback accepted"})
		})
		api.GET("/routing-model/status", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"status": "ready", "backend": "fixture"})
		})
		api.DELETE("/calls", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"message": "all calls deleted"})
		})
		api.DELETE("/calls/:id", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"message": "call deleted", "id": c.Param("id")})
		})
		api.GET("/audit/events", func(c *gin.Context) {
			c.JSON(http.StatusOK, gin.H{"items": []gin.H{}})
		})
	}

	port := os.Getenv("DAST_FIXTURE_PORT")
	if strings.TrimSpace(port) == "" {
		port = "8013"
	}

	log.Printf("Starting lightweight DAST fixture on http://127.0.0.1:%s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatalf("run DAST fixture: %v", err)
	}
}

func mockAuth() gin.HandlerFunc {
	return func(c *gin.Context) {
		token := strings.TrimSpace(c.GetHeader("Authorization"))
		if token != "Bearer "+fixtureToken {
			c.AbortWithStatusJSON(http.StatusUnauthorized, gin.H{"error": "missing or invalid bearer token"})
			return
		}
		c.Next()
	}
}

func fixtureWebDir() (string, error) {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return "", os.ErrNotExist
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "web")), nil
}
