package brain

import (
	"database/sql"
	"strings"
)

// RecallConcepts fetches a few high-importance concepts matching a topic.
// This is a lightweight LTM recall: no embeddings, just LIKE.
func RecallConcepts(db *sql.DB, topic string, limit int) string {
	if db == nil {
		return ""
	}
	topic = strings.TrimSpace(topic)
	if topic == "" {
		return ""
	}
	if limit <= 0 {
		limit = 3
	}
	if limit > 8 {
		limit = 8
	}
	pat := "%" + topic + "%"

	rows, err := db.Query(
		`SELECT term, summary, confidence, importance
		 FROM concepts
		 WHERE term LIKE ? OR summary LIKE ?
		 ORDER BY importance DESC, confidence DESC
		 LIMIT ?`,
		pat, pat, limit,
	)
	if err != nil {
		return ""
	}
	defer rows.Close()

	var b strings.Builder
	for rows.Next() {
		var term, sum string
		var conf, imp float64
		if err := rows.Scan(&term, &sum, &conf, &imp); err != nil {
			continue
		}
		term = strings.TrimSpace(term)
		sum = strings.TrimSpace(sum)
		if term == "" || sum == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(term)
		b.WriteString(": ")
		b.WriteString(clipForContext(sum, 240))
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}
