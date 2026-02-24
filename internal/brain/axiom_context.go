package brain

import (
	"database/sql"
	"fmt"
	"strings"
	"time"
)

type AxiomInterp struct {
	AxiomID     int
	Kind        string
	Key         string
	Value       string
	Confidence  float64
	SourceNote  string
	UpdatedAt   string
}

func ensureAxiomInterpretationsTable(db *sql.DB) {
	if db == nil {
		return
	}
	_, _ = db.Exec(`
CREATE TABLE IF NOT EXISTS axiom_interpretations(
  axiom_id INTEGER NOT NULL,
  kind TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.5,
  source_note TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  UNIQUE(axiom_id, kind, key)
);`)
}

func ListAxiomInterpretations(db *sql.DB, axiomID int, limit int) ([]AxiomInterp, error) {
	if db == nil {
		return nil, nil
	}
	ensureAxiomInterpretationsTable(db)
	if axiomID < 1 || axiomID > 4 {
		axiomID = 1
	}
	if limit <= 0 {
		limit = 10
	}
	rows, err := db.Query(`
SELECT axiom_id, kind, key, value, confidence, source_note, updated_at
FROM axiom_interpretations
WHERE axiom_id=?
ORDER BY
  CASE kind WHEN 'rule' THEN 0 WHEN 'metric' THEN 1 WHEN 'definition' THEN 2 ELSE 3 END,
  confidence DESC,
  updated_at DESC
LIMIT ?`, axiomID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []AxiomInterp{}
	for rows.Next() {
		var it AxiomInterp
		_ = rows.Scan(&it.AxiomID, &it.Kind, &it.Key, &it.Value, &it.Confidence, &it.SourceNote, &it.UpdatedAt)
		out = append(out, it)
	}
	return out, nil
}

func RenderAxiomInterpretations(db *sql.DB, axiomID int, limit int) string {
	if db == nil {
		return "DB missing."
	}
	items, err := ListAxiomInterpretations(db, axiomID, limit)
	if err != nil || len(items) == 0 {
		return "Keine Interpretationen gefunden."
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("axiom_interpretations (A%d, newest/strongest first):\n", axiomID))
	for _, it := range items {
		src := strings.TrimSpace(it.SourceNote)
		if src != "" {
			src = " (" + src + ")"
		}
		b.WriteString(fmt.Sprintf("- %s:%s = %s [c=%.2f]%s\n", it.Kind, it.Key, it.Value, it.Confidence, src))
	}
	return strings.TrimSpace(b.String())
}

func RenderAxiomContext(db *sql.DB, perAxiom int) string {
	if db == nil {
		return ""
	}
	if perAxiom <= 0 {
		perAxiom = 1
	}
	ensureAxiomInterpretationsTable(db)
	var b strings.Builder
	b.WriteString("AXIOM_CONTEXT (ops-hints):\n")
	any := false
	for _, ax := range KernelAxioms {
		items, _ := ListAxiomInterpretations(db, ax.ID, perAxiom)
		if len(items) == 0 {
			continue
		}
		any = true
		b.WriteString(fmt.Sprintf("A%d: ", ax.ID))
		for i, it := range items {
			if i > 0 {
				b.WriteString(" | ")
			}
			// keep very short
			val := strings.TrimSpace(it.Value)
			if len(val) > 160 {
				val = val[:160] + "..."
			}
			b.WriteString(fmt.Sprintf("%s:%s=%s", it.Kind, it.Key, val))
		}
		b.WriteString("\n")
	}
	if !any {
		return ""
	}
	_ = time.Now() // reserved for later (context freshness)
	return strings.TrimSpace(b.String())
}

func ApplyAxiomContextToUserText(ws *Workspace, userText string) string {
	if ws == nil {
		return userText
	}
	ctx := strings.TrimSpace(ws.AxiomContext)
	if ctx == "" {
		return userText
	}
	// Only affects the LLM prompt; user never sees this wrapper directly.
	return ctx + "\n\nUSER:\n" + strings.TrimSpace(userText)
}
