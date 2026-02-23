package brain

import (
	"database/sql"
	"strings"
)

// RecentThoughtSnippets returns a compact, high-signal snippet list of recent internal events
// (daydream/thought) to help the daydreamer produce a more human, drifting inner monologue.
func RecentThoughtSnippets(db *sql.DB, topic string, k int) string {
	if db == nil {
		return ""
	}
	topic = strings.TrimSpace(topic)
	if k <= 0 {
		k = 6
	}
	if k > 16 {
		k = 16
	}

	// Prefer topic-matched recent events; fall back to global.
	rows, err := db.Query(`SELECT kind, topic, details FROM events WHERE kind IN ('daydream','thought') AND topic=? ORDER BY id DESC LIMIT ?`, topic, k)
	if err != nil {
		return ""
	}
	defer rows.Close()

	var b strings.Builder
	for rows.Next() {
		var kind, tp, det string
		if rows.Scan(&kind, &tp, &det) != nil {
			continue
		}
		kind = strings.TrimSpace(kind)
		det = strings.TrimSpace(det)
		if det == "" {
			continue
		}
		b.WriteString("- ")
		if kind != "" {
			b.WriteString(kind)
			b.WriteString(": ")
		}
		b.WriteString(clipForContext(det, 180))
		b.WriteString("\n")
	}
	out := strings.TrimSpace(b.String())
	if out != "" {
		return out
	}

	rows2, err2 := db.Query(`SELECT kind, topic, details FROM events WHERE kind IN ('daydream','thought') ORDER BY id DESC LIMIT ?`, k)
	if err2 != nil {
		return ""
	}
	defer rows2.Close()
	b.Reset()
	for rows2.Next() {
		var kind, tp, det string
		if rows2.Scan(&kind, &tp, &det) != nil {
			continue
		}
		det = strings.TrimSpace(det)
		if det == "" {
			continue
		}
		b.WriteString("- ")
		if tp != "" {
			b.WriteString("[")
			b.WriteString(tp)
			b.WriteString("] ")
		}
		b.WriteString(clipForContext(det, 180))
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}
