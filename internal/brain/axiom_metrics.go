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
