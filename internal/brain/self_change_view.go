package brain

import (
	"database/sql"
	"fmt"
	"strings"
)

func RenderSelfChanges(db *sql.DB, kind string, limit int) string {
	if db == nil {
		return "DB missing."
	}
	if limit <= 0 {
		limit = 20
	}
	kind = strings.TrimSpace(kind)
	q := `
SELECT created_at, kind, target, axiom_goal, allowed, axiom_block, risk, energy_cost, note
FROM self_changes
WHERE (?='' OR kind=?)
ORDER BY created_at DESC
LIMIT ?`
	rows, err := db.Query(q, kind, kind, limit)
	if err != nil {
		return "ERR: " + err.Error()
	}
	defer rows.Close()
	var b strings.Builder
	b.WriteString("self_changes (newest first):\n")
	n := 0
	for rows.Next() {
		var createdAt, k, target, risk, note string
		var axGoal, allowed, axBlock int
		var energy float64
		_ = rows.Scan(&createdAt, &k, &target, &axGoal, &allowed, &axBlock, &risk, &energy, &note)
		n++
		b.WriteString(fmt.Sprintf("- %s kind=%s target=%s ax=%d allowed=%d block=%d risk=%s cost=%.2f note=%s\n",
			createdAt, k, target, axGoal, allowed, axBlock, risk, energy, strings.TrimSpace(note)))
	}
	if n == 0 {
		return "Keine self_changes."
	}
	return strings.TrimSpace(b.String())
}
}
