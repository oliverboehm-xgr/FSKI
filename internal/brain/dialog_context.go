package brain

import (
	"database/sql"
	"strings"
)

type Turn struct {
	Kind string
	Text string
}

// BuildDialogContext returns the last N dialog turns (user + bunny) as plain text.
// Uses messages + message_meta(kind). We intentionally exclude "think" (internal).
func BuildDialogContext(db *sql.DB, limit int) string {
	if db == nil || limit <= 0 {
		return ""
	}
	if limit > 40 {
		limit = 40
	}

	rows, err := db.Query(
		`SELECT COALESCE(mm.kind,'reply') AS kind, m.text
		 FROM messages m
		 LEFT JOIN message_meta mm ON mm.message_id = m.id
		 WHERE COALESCE(mm.kind,'reply') IN ('user','reply','auto')
		 ORDER BY m.id DESC
		 LIMIT ?`,
		limit,
	)
	if err != nil {
		return ""
	}
	defer rows.Close()

	var rev []Turn
	for rows.Next() {
		var k, t string
		if err := rows.Scan(&k, &t); err != nil {
			continue
		}
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		rev = append(rev, Turn{Kind: k, Text: clipForContext(t, 500)})
	}
	if len(rev) == 0 {
		return ""
	}

	// reverse into chronological order
	var b strings.Builder
	for i := len(rev) - 1; i >= 0; i-- {
		role := "Bunny"
		if rev[i].Kind == "user" {
			role = "User"
		}
		b.WriteString(role)
		b.WriteString(": ")
		b.WriteString(rev[i].Text)
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}

func clipForContext(s string, n int) string {
	if n <= 0 {
		return s
	}
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
