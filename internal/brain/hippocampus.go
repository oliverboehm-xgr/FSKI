package brain

import (
	"database/sql"
	"strings"
	"time"
)

type ConsolidateRequest struct {
	Topic      string
	StartEvent int64
	EndEvent   int64
	TextBlock  string
}

// NeedsConsolidation: every N events, create/update an episode gist.
func NeedsConsolidation(db *sql.DB, egAny any, topic string) (bool, ConsolidateRequest) {
	eg, ok := egAny.(interface {
		MemoryParams() (int, int, int, float64, float64, int, bool)
	})
	if db == nil || !ok || strings.TrimSpace(topic) == "" {
		return false, ConsolidateRequest{}
	}
	conEvery, _, _, _, _, _, _ := eg.MemoryParams()

	var lastEnd int64
	_ = db.QueryRow(`SELECT COALESCE(MAX(end_event_id),0) FROM episodes WHERE topic=?`, topic).Scan(&lastEnd)

	var newest int64
	_ = db.QueryRow(`SELECT COALESCE(MAX(id),0) FROM events WHERE topic=?`, topic).Scan(&newest)
	if newest <= 0 {
		return false, ConsolidateRequest{}
	}

	if newest-lastEnd < int64(conEvery) {
		return false, ConsolidateRequest{}
	}

	start := lastEnd + 1
	end := newest

	rows, err := db.Query(
		`SELECT channel, text FROM events WHERE topic=? AND id BETWEEN ? AND ? ORDER BY id ASC LIMIT 60`,
		topic, start, end,
	)
	if err != nil {
		return false, ConsolidateRequest{}
	}
	defer rows.Close()

	var b strings.Builder
	for rows.Next() {
		var ch, t string
		_ = rows.Scan(&ch, &t)
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		b.WriteString(ch)
		b.WriteString(": ")
		b.WriteString(clipForContext(t, 420))
		b.WriteString("\n")
	}
	tb := strings.TrimSpace(b.String())
	if tb == "" {
		return false, ConsolidateRequest{}
	}

	return true, ConsolidateRequest{
		Topic:      topic,
		StartEvent: start,
		EndEvent:   end,
		TextBlock:  tb,
	}
}

func SaveEpisode(db *sql.DB, topic string, start, end int64, summary string) {
	if db == nil || strings.TrimSpace(topic) == "" || strings.TrimSpace(summary) == "" {
		return
	}
	_, _ = db.Exec(
		`INSERT INTO episodes(created_at, topic, start_event_id, end_event_id, summary, salience)
         VALUES(?,?,?,?,?,?)`,
		time.Now().Format(time.RFC3339), topic, start, end, summary, 0.65,
	)
}
