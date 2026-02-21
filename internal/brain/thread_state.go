package brain

import (
	"database/sql"
	"time"
)

func LoadActiveTopic(db *sql.DB) string {
	if db == nil {
		return ""
	}
	var v string
	_ = db.QueryRow(`SELECT value FROM thread_state WHERE key='active_topic'`).Scan(&v)
	return v
}

func SaveActiveTopic(db *sql.DB, topic string) {
	if db == nil {
		return
	}
	if topic == "" {
		return
	}
	_, _ = db.Exec(
		`INSERT INTO thread_state(key,value,updated_at) VALUES('active_topic',?,?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		topic, time.Now().Format(time.RFC3339),
	)
}
