package tests

import (
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"
	"orchestrator/internal/clients"
	callprocessingv1 "orchestrator/internal/gen"
	"orchestrator/internal/services"
)

type fakeTranscriptionServer struct {
	callprocessingv1.UnimplementedTranscriptionServiceServer
	mu       sync.Mutex
	requests []*callprocessingv1.TranscribeRequest
	response *callprocessingv1.Transcript
}

func (s *fakeTranscriptionServer) Transcribe(_ context.Context, req *callprocessingv1.TranscribeRequest) (*callprocessingv1.TranscribeResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	return &callprocessingv1.TranscribeResponse{
		Transcript: s.response,
	}, nil
}

func (s *fakeTranscriptionServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

type fakeRoutingServer struct {
	callprocessingv1.UnimplementedRoutingServiceServer
	mu       sync.Mutex
	requests []*callprocessingv1.RouteRequest
	response *callprocessingv1.Routing
}

func (s *fakeRoutingServer) Route(_ context.Context, req *callprocessingv1.RouteRequest) (*callprocessingv1.RouteResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	return &callprocessingv1.RouteResponse{
		Routing: s.response,
	}, nil
}

func (s *fakeRoutingServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

type fakeTicketServer struct {
	callprocessingv1.UnimplementedTicketServiceServer
	mu       sync.Mutex
	requests []*callprocessingv1.CreateTicketRequest
	response *callprocessingv1.TicketCreated
}

func (s *fakeTicketServer) CreateTicket(_ context.Context, req *callprocessingv1.CreateTicketRequest) (*callprocessingv1.CreateTicketResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	return &callprocessingv1.CreateTicketResponse{
		Ticket: s.response,
	}, nil
}

func (s *fakeTicketServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

func (s *fakeTicketServer) lastRequest() *callprocessingv1.CreateTicketRequest {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.requests) == 0 {
		return nil
	}
	return s.requests[len(s.requests)-1]
}

type fakeNotificationServer struct {
	callprocessingv1.UnimplementedNotificationServiceServer
	mu       sync.Mutex
	requests []*callprocessingv1.SendNotificationRequest
	response *callprocessingv1.SendNotificationResponse
}

func (s *fakeNotificationServer) SendNotification(_ context.Context, req *callprocessingv1.SendNotificationRequest) (*callprocessingv1.SendNotificationResponse, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, req)
	return s.response, nil
}

func (s *fakeNotificationServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

func startGRPCServer(t *testing.T, register func(*grpc.Server)) string {
	t.Helper()

	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	server := grpc.NewServer()
	register(server)

	go func() {
		_ = server.Serve(listener)
	}()

	t.Cleanup(func() {
		server.Stop()
		_ = listener.Close()
	})

	return listener.Addr().String()
}

func writeAudioFile(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "call.wav")
	if err := os.WriteFile(path, []byte("fake audio bytes"), 0o644); err != nil {
		t.Fatalf("write audio file: %v", err)
	}
	return path
}

func TestProcessCallIntegrationFullPipeline(t *testing.T) {
	transcriptMeta, err := structpb.NewStruct(map[string]interface{}{
		"agent": "operator-1",
	})
	if err != nil {
		t.Fatalf("build metadata: %v", err)
	}

	transcriptionServer := &fakeTranscriptionServer{
		response: &callprocessingv1.Transcript{
			CallId: "call-001",
			Segments: []*callprocessingv1.Segment{
				{Start: 0, End: 2, Speaker: "spk_0", Role: "agent", Text: "Здравствуйте, чем могу помочь?"},
				{Start: 2, End: 5, Speaker: "spk_1", Role: "caller", Text: "У меня проблема с заказом 12345."},
			},
			RoleMapping: map[string]string{
				"spk_0": "agent",
				"spk_1": "caller",
			},
			Metadata: transcriptMeta,
		},
	}

	routingServer := &fakeRoutingServer{
		response: &callprocessingv1.Routing{
			IntentId:         "orders.problem",
			IntentConfidence: 0.91,
			Priority:         "high",
			SuggestedGroup:   "support",
			SpamCheck: &callprocessingv1.SpamCheck{
				Status:         "allow",
				PredictedLabel: "not_spam",
				Confidence:     0.99,
			},
		},
	}

	ticketServer := &fakeTicketServer{
		response: &callprocessingv1.TicketCreated{
			TicketId:   "ticket-001",
			ExternalId: "EXT-001",
			Url:        "http://ticket.local/ticket-001",
			System:     "mock",
			CreatedAt:  timestamppb.New(time.Now()),
		},
	}

	notificationServer := &fakeNotificationServer{
		response: &callprocessingv1.SendNotificationResponse{
			Success: true,
			Results: []*callprocessingv1.NotificationChannel{
				{Type: "email", Success: true, Destination: "support@example.local"},
			},
		},
	}

	entityCalls := 0
	entityServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		entityCalls++
		if r.URL.Path != "/api/extract-entities" {
			t.Fatalf("unexpected path: %s", r.URL.Path)
		}
		var req map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode entity request: %v", err)
		}
		segments, ok := req["segments"].([]interface{})
		if !ok || len(segments) != 2 {
			t.Fatalf("unexpected entity request segments: %#v", req["segments"])
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"entities": {
				"persons": [],
				"phones": [],
				"emails": [],
				"order_ids": [{"type":"order_id","value":"12345","confidence":0.97,"context":"проблема с заказом 12345"}],
				"account_ids": [],
				"money_amounts": [],
				"dates": []
			}
		}`))
	}))
	defer entityServer.Close()

	transcriptionAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterTranscriptionServiceServer(server, transcriptionServer)
	})
	routingAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterRoutingServiceServer(server, routingServer)
	})
	ticketAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterTicketServiceServer(server, ticketServer)
	})
	notificationAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterNotificationServiceServer(server, notificationServer)
	})

	transcriptionClient, err := clients.NewTranscriptionClient(transcriptionAddr)
	if err != nil {
		t.Fatalf("new transcription client: %v", err)
	}
	defer transcriptionClient.Close()

	routingClient, err := clients.NewRoutingClient(routingAddr)
	if err != nil {
		t.Fatalf("new routing client: %v", err)
	}
	defer routingClient.Close()

	ticketClient, err := clients.NewTicketClient(ticketAddr, 10*time.Second)
	if err != nil {
		t.Fatalf("new ticket client: %v", err)
	}
	defer ticketClient.Close()

	notificationClient, err := clients.NewNotificationClient(notificationAddr)
	if err != nil {
		t.Fatalf("new notification client: %v", err)
	}
	defer notificationClient.Close()

	service := services.NewOrchestratorService(
		transcriptionClient,
		routingClient,
		ticketClient,
		notificationClient,
		clients.NewEntityClient(entityServer.URL),
		0.5,
	)

	result, err := service.ProcessCall(writeAudioFile(t))
	if err != nil {
		t.Fatalf("process call: %v", err)
	}

	if result.Status != services.ProcessStatusCompleted {
		t.Fatalf("expected status %q, got %q", services.ProcessStatusCompleted, result.Status)
	}
	if result.CallID != "call-001" {
		t.Fatalf("unexpected call id: %s", result.CallID)
	}
	if transcriptionServer.count() != 1 {
		t.Fatalf("expected 1 transcription call, got %d", transcriptionServer.count())
	}
	if routingServer.count() != 1 {
		t.Fatalf("expected 1 routing call, got %d", routingServer.count())
	}
	if ticketServer.count() != 1 {
		t.Fatalf("expected 1 ticket call, got %d", ticketServer.count())
	}
	if notificationServer.count() != 1 {
		t.Fatalf("expected 1 notification call, got %d", notificationServer.count())
	}
	if entityCalls != 1 {
		t.Fatalf("expected 1 entity call, got %d", entityCalls)
	}
	if result.Ticket == nil || result.Ticket.TicketID != "ticket-001" {
		t.Fatalf("unexpected ticket in result: %#v", result.Ticket)
	}
	if result.Notification == nil || !result.Notification.Success {
		t.Fatalf("unexpected notification result: %#v", result.Notification)
	}
	if result.Entities == nil || len(result.Entities.OrderIDs) != 1 {
		t.Fatalf("unexpected entities in result: %#v", result.Entities)
	}
	if result.ProcessingTime["transcription"] <= 0 {
		t.Fatalf("transcription time must be > 0")
	}
	if result.ProcessingTime["routing"] <= 0 {
		t.Fatalf("routing time must be > 0")
	}
	if result.ProcessingTime["entity_extraction"] <= 0 {
		t.Fatalf("entity extraction time must be > 0")
	}
	if result.ProcessingTime["ticket_creation"] <= 0 {
		t.Fatalf("ticket creation time must be > 0")
	}
	if result.ProcessingTime["notification"] <= 0 {
		t.Fatalf("notification time must be > 0")
	}

	ticketReq := ticketServer.lastRequest()
	if ticketReq == nil {
		t.Fatalf("ticket request was not captured")
	}
	if ticketReq.GetTranscript().GetCallId() != "call-001" {
		t.Fatalf("unexpected call id in ticket request: %s", ticketReq.GetTranscript().GetCallId())
	}
	if ticketReq.GetRouting().GetIntentId() != "orders.problem" {
		t.Fatalf("unexpected intent in ticket request: %s", ticketReq.GetRouting().GetIntentId())
	}
	if len(ticketReq.GetEntities().GetOrderIds()) != 1 {
		t.Fatalf("unexpected order ids in ticket request: %#v", ticketReq.GetEntities().GetOrderIds())
	}
}

func TestProcessCallIntegrationStopsOnLowConfidenceRouting(t *testing.T) {
	transcriptionServer := &fakeTranscriptionServer{
		response: &callprocessingv1.Transcript{
			CallId: "call-002",
			Segments: []*callprocessingv1.Segment{
				{Start: 0, End: 1, Speaker: "spk_0", Role: "caller", Text: "Мне нужна помощь"},
			},
			RoleMapping: map[string]string{
				"spk_0": "caller",
			},
		},
	}

	routingServer := &fakeRoutingServer{
		response: &callprocessingv1.Routing{
			IntentId:         "misc.triage",
			IntentConfidence: 0.22,
			Priority:         "medium",
			SuggestedGroup:   "triage",
			SpamCheck: &callprocessingv1.SpamCheck{
				Status:         "allow",
				PredictedLabel: "not_spam",
				Confidence:     0.9,
			},
		},
	}

	ticketServer := &fakeTicketServer{
		response: &callprocessingv1.TicketCreated{
			TicketId: "ticket-002",
			Url:      "http://ticket.local/ticket-002",
			System:   "mock",
			CreatedAt: timestamppb.New(time.Now()),
		},
	}

	notificationServer := &fakeNotificationServer{
		response: &callprocessingv1.SendNotificationResponse{Success: true},
	}

	entityCalls := 0
	entityServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		entityCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"entities":{}}`))
	}))
	defer entityServer.Close()

	transcriptionAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterTranscriptionServiceServer(server, transcriptionServer)
	})
	routingAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterRoutingServiceServer(server, routingServer)
	})
	ticketAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterTicketServiceServer(server, ticketServer)
	})
	notificationAddr := startGRPCServer(t, func(server *grpc.Server) {
		callprocessingv1.RegisterNotificationServiceServer(server, notificationServer)
	})

	transcriptionClient, err := clients.NewTranscriptionClient(transcriptionAddr)
	if err != nil {
		t.Fatalf("new transcription client: %v", err)
	}
	defer transcriptionClient.Close()

	routingClient, err := clients.NewRoutingClient(routingAddr)
	if err != nil {
		t.Fatalf("new routing client: %v", err)
	}
	defer routingClient.Close()

	ticketClient, err := clients.NewTicketClient(ticketAddr, 10*time.Second)
	if err != nil {
		t.Fatalf("new ticket client: %v", err)
	}
	defer ticketClient.Close()

	notificationClient, err := clients.NewNotificationClient(notificationAddr)
	if err != nil {
		t.Fatalf("new notification client: %v", err)
	}
	defer notificationClient.Close()

	service := services.NewOrchestratorService(
		transcriptionClient,
		routingClient,
		ticketClient,
		notificationClient,
		clients.NewEntityClient(entityServer.URL),
		0.5,
	)

	result, err := service.ProcessCall(writeAudioFile(t))
	if err != nil {
		t.Fatalf("process call: %v", err)
	}

	if result.Status != services.ProcessStatusAwaitingRoutingReview {
		t.Fatalf("expected status %q, got %q", services.ProcessStatusAwaitingRoutingReview, result.Status)
	}
	if ticketServer.count() != 0 {
		t.Fatalf("ticket service must not be called, got %d calls", ticketServer.count())
	}
	if notificationServer.count() != 0 {
		t.Fatalf("notification service must not be called, got %d calls", notificationServer.count())
	}
	if entityCalls != 0 {
		t.Fatalf("entity service must not be called, got %d calls", entityCalls)
	}
	if result.Ticket != nil {
		t.Fatalf("ticket must be nil when routing review is required")
	}
	if result.Notification != nil {
		t.Fatalf("notification must be nil when routing review is required")
	}
}
