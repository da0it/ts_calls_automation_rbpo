package handlers

import (
	"bytes"
	"encoding/json"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestProcessCallRejectsMissingAudio(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	handler := &ProcessHandler{}
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/process-call", bytes.NewReader(nil))
	req.Header.Set("Content-Type", "multipart/form-data; boundary=empty")
	ctx.Request = req

	handler.ProcessCall(ctx)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, recorder.Code)
	}

	var body map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body["error"] != "audio file is required" {
		t.Fatalf("unexpected error message: %q", body["error"])
	}
}

func TestProcessCallRejectsUnsupportedExtension(t *testing.T) {
	t.Parallel()
	gin.SetMode(gin.TestMode)

	handler := &ProcessHandler{}
	recorder := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(recorder)

	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	part, err := writer.CreateFormFile("audio", "sample.txt")
	if err != nil {
		t.Fatalf("create form file: %v", err)
	}
	if _, err := part.Write([]byte("not audio")); err != nil {
		t.Fatalf("write payload: %v", err)
	}
	if err := writer.Close(); err != nil {
		t.Fatalf("close writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/process-call", &body)
	req.Header.Set("Content-Type", writer.FormDataContentType())
	ctx.Request = req

	handler.ProcessCall(ctx)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("expected status %d, got %d", http.StatusBadRequest, recorder.Code)
	}

	var resp map[string]string
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !strings.Contains(resp["error"], "unsupported audio format") {
		t.Fatalf("unexpected error message: %q", resp["error"])
	}
}
