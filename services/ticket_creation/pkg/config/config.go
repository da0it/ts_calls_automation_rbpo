// pkg/config/config.go
package config

import (
	"os"
	"strconv"
	"strings"
)

type Config struct {
	// Server
	ServerPort         string
	GRPCPort           string
	GRPCTLSEnabled     bool
	GRPCTLSCertFile    string
	GRPCTLSKeyFile     string
	CORSAllowedOrigins string

	// Database
	DatabaseURL string

	// Python services
	PythonNERServiceURL string

	// LLM
	LLMProvider              string
	LLMRequestTimeoutSeconds int
	OllamaBaseURL            string
	OllamaModel              string
	OllamaTemperature        float64
	OllamaNumPredict         int
	AnthropicAPIKey          string
	AnthropicModel           string

	// Ticket systems
	TicketSystem  string //so, mock
	JiraURL       string
	JiraUser      string
	JiraAPIToken  string
	RedmineURL    string
	RedmineAPIKey string

	TicketIncludePIIInDescription bool
}

func Load() *Config {
	return &Config{
		ServerPort:                    getEnv("SERVER_PORT", "8080"),
		GRPCPort:                      getEnv("GRPC_PORT", "50054"),
		GRPCTLSEnabled:                getEnvBool("TICKET_GRPC_TLS_ENABLED", false),
		GRPCTLSCertFile:               getEnv("TICKET_GRPC_TLS_CERT_FILE", ""),
		GRPCTLSKeyFile:                getEnv("TICKET_GRPC_TLS_KEY_FILE", ""),
		CORSAllowedOrigins:            getEnv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:3000"),
		DatabaseURL:                   getEnv("DATABASE_URL", "postgres://localhost/tickets?sslmode=disable"),
		PythonNERServiceURL:           getEnv("PYTHON_NER_SERVICE_URL", "http://localhost:5000"),
		LLMProvider:                   getEnv("LLM_PROVIDER", "ollama"),
		LLMRequestTimeoutSeconds:      getEnvInt("LLM_REQUEST_TIMEOUT_SECONDS", 60),
		OllamaBaseURL:                 getEnv("OLLAMA_BASE_URL", "http://localhost:11434"),
		OllamaModel:                   getEnv("OLLAMA_MODEL", "qwen2.5:7b"),
		OllamaTemperature:             getEnvFloat("OLLAMA_TEMPERATURE", 0.2),
		OllamaNumPredict:              getEnvInt("OLLAMA_NUM_PREDICT", 512),
		AnthropicAPIKey:               getEnv("ANTHROPIC_API_KEY", ""),
		AnthropicModel:                getEnv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
		TicketSystem:                  getEnv("TICKET_SYSTEM", "mock"),
		TicketIncludePIIInDescription: getEnvBool("TICKET_INCLUDE_PII_IN_DESCRIPTION", false),
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intVal, err := strconv.Atoi(value); err == nil {
			return intVal
		}
	}
	return defaultValue
}

func getEnvFloat(key string, defaultValue float64) float64 {
	if value := os.Getenv(key); value != "" {
		if floatVal, err := strconv.ParseFloat(value, 64); err == nil {
			return floatVal
		}
	}
	return defaultValue
}

func getEnvBool(key string, defaultValue bool) bool {
	value := strings.TrimSpace(strings.ToLower(getEnv(key, "")))
	if value == "" {
		return defaultValue
	}
	switch value {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return defaultValue
	}
}
