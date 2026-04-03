package services

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

type CallQueueService struct {
	db *sql.DB
}

func NewCallQueueService(db *sql.DB) *CallQueueService {
	return &CallQueueService{db: db}
}

func (s *CallQueueService) Migrate() error {
	query := `
	CREATE TABLE IF NOT EXISTS call_queue (
		id VARCHAR(128) PRIMARY KEY,
		payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
		created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
		updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
	);
	CREATE INDEX IF NOT EXISTS idx_call_queue_created_at ON call_queue (created_at DESC);
	`
	_, err := s.db.Exec(query)
	return err
}

func (s *CallQueueService) Save(payload map[string]interface{}) (map[string]interface{}, error) {
	if payload == nil {
		return nil, fmt.Errorf("call payload is required")
	}

	id := strings.TrimSpace(asString(payload["id"]))
	if id == "" {
		id = fmt.Sprintf("call-%d", time.Now().UnixNano())
		payload["id"] = id
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("marshal call payload: %w", err)
	}

	_, err = s.db.Exec(
		`INSERT INTO call_queue (id, payload_json, created_at, updated_at)
		 VALUES ($1, $2::jsonb, NOW(), NOW())
		 ON CONFLICT (id) DO UPDATE SET
		   payload_json = EXCLUDED.payload_json,
		   updated_at = NOW()`,
		id,
		string(raw),
	)
	if err != nil {
		return nil, fmt.Errorf("save call payload: %w", err)
	}

	return payload, nil
}

func (s *CallQueueService) List(limit, offset int) ([]map[string]interface{}, error) {
	if limit <= 0 {
		limit = 200
	}
	if limit > 1000 {
		limit = 1000
	}
	if offset < 0 {
		offset = 0
	}

	rows, err := s.db.Query(
		`SELECT payload_json
		 FROM call_queue
		 ORDER BY created_at DESC, id DESC
		 LIMIT $1 OFFSET $2`,
		limit,
		offset,
	)
	if err != nil {
		return nil, fmt.Errorf("query call queue: %w", err)
	}
	defer rows.Close()

	calls := make([]map[string]interface{}, 0, limit)
	for rows.Next() {
		var raw []byte
		if err := rows.Scan(&raw); err != nil {
			return nil, fmt.Errorf("scan call queue: %w", err)
		}

		payload := map[string]interface{}{}
		if len(raw) > 0 {
			if err := json.Unmarshal(raw, &payload); err != nil {
				return nil, fmt.Errorf("decode call payload: %w", err)
			}
		}
		calls = append(calls, payload)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterate call queue: %w", err)
	}

	return calls, nil
}

func (s *CallQueueService) Get(id string) (map[string]interface{}, error) {
	id = strings.TrimSpace(id)
	if id == "" {
		return nil, fmt.Errorf("call id is required")
	}

	var raw []byte
	if err := s.db.QueryRow(`SELECT payload_json FROM call_queue WHERE id = $1`, id).Scan(&raw); err != nil {
		if err == sql.ErrNoRows {
			return nil, err
		}
		return nil, fmt.Errorf("get call queue item: %w", err)
	}

	payload := map[string]interface{}{}
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &payload); err != nil {
			return nil, fmt.Errorf("decode call payload: %w", err)
		}
	}
	return payload, nil
}

func (s *CallQueueService) Delete(id string) error {
	id = strings.TrimSpace(id)
	if id == "" {
		return fmt.Errorf("call id is required")
	}

	result, err := s.db.Exec(`DELETE FROM call_queue WHERE id = $1`, id)
	if err != nil {
		return fmt.Errorf("delete call: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("delete call rows affected: %w", err)
	}
	if affected == 0 {
		return sql.ErrNoRows
	}
	return nil
}

func (s *CallQueueService) Clear() (int64, error) {
	result, err := s.db.Exec(`DELETE FROM call_queue`)
	if err != nil {
		return 0, fmt.Errorf("clear call queue: %w", err)
	}
	affected, err := result.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("clear call queue rows affected: %w", err)
	}
	return affected, nil
}

func asString(value interface{}) string {
	if value == nil {
		return ""
	}
	switch v := value.(type) {
	case string:
		return v
	default:
		return fmt.Sprintf("%v", value)
	}
}
