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
diff --git a/internal/brain/axiom_metrics.go b/internal/brain/axiom_metrics.go
new file mode 100644
index 0000000000000000000000000000000000000000..bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
--- /dev/null
 b/internal/brain/axiom_metrics.go
@@ -0,0 +1,175 @@
package brain

import (
	"database/sql"
	"fmt"
	"sort"
	"strings"
	"time"
)

func ensureAxiomMetricsTable(db *sql.DB) {
	if db == nil {
		return
	}
	_, _ = db.Exec(`
CREATE TABLE IF NOT EXISTS axiom_metrics(
  key TEXT PRIMARY KEY,
  value REAL NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT ''
);`)
}

func SetAxiomMetric(db *sql.DB, key string, value float64, note string) {
	if db == nil {
		return
	}
	ensureAxiomMetricsTable(db)
	key = strings.TrimSpace(key)
	if key == "" {
		return
	}
	now := time.Now().Format(time.RFC3339)
	_, _ = db.Exec(
		`INSERT INTO axiom_metrics(key,value,updated_at,note) VALUES(?,?,?,?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at, note=excluded.note`,
		key, value, now, strings.TrimSpace(note),
	)
}

func ListAxiomMetrics(db *sql.DB, limit int) map[string]float64 {
	if db == nil {
		return map[string]float64{}
	}
	ensureAxiomMetricsTable(db)
	if limit <= 0 {
		limit = 50
	}
	rows, err := db.Query(`SELECT key,value FROM axiom_metrics ORDER BY updated_at DESC LIMIT ?`, limit)
	if err != nil {
		return map[string]float64{}
	}
	defer rows.Close()
	out := map[string]float64{}
	for rows.Next() {
		var k string
		var v float64
		_ = rows.Scan(&k, &v)
		out[strings.TrimSpace(k)] = v
	}
	return out
}

func RenderAxiomMetrics(db *sql.DB, limit int) string {
	m := ListAxiomMetrics(db, limit)
	if len(m) == 0 {
		return "Keine axiom_metrics."
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	var b strings.Builder
	b.WriteString("axiom_metrics:\n")
	for _, k := range keys {
		b.WriteString(fmt.Sprintf("- %s = %.4f\n", k, m[k]))
	}
	return strings.TrimSpace(b.String())
}

// AugmentPolicyContextWithAxiomMetrics turns a few tracked metrics into discrete buckets
// so the bandit can actually learn different posteriors per regime.
func AugmentPolicyContextWithAxiomMetrics(db *sql.DB, ctxKey string) string {
	if db == nil || strings.TrimSpace(ctxKey) == "" {
		return ctxKey
	}
	turns := kvInt(db, "metric:turns", 0)
	if turns <= 0 {
		return ctxKey
	}
	research := kvInt(db, "metric:action:research_then_answer", 0)
	ratio := float64(research) / float64(turns)

	bin := "ev=lo"
	if ratio >= 0.40 {
		bin = "ev=hi"
	} else if ratio >= 0.20 {
		bin = "ev=med"
	}

	// Persist as metric as well (for UI / debugging).
	SetAxiomMetric(db, "evidence_ratio", ratio, "derived: research_then_answer / turns")
	return ctxKey + "|" + bin
}
