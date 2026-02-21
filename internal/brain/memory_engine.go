package brain

import (
	"database/sql"
	"math"
	"sort"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
)

// InsertEvent stores a generic event (multi-channel).
func InsertEvent(db *sql.DB, channel, topic, text string, messageID int64, salience float64) {
	if db == nil {
		return
	}
	channel = strings.TrimSpace(channel)
	if channel == "" {
		channel = "unknown"
	}
	topic = strings.TrimSpace(topic)
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}
	if salience < 0 {
		salience = 0
	}
	if salience > 1 {
		salience = 1
	}
	var mid any = nil
	if messageID > 0 {
		mid = messageID
	}
	_, _ = db.Exec(
		`INSERT INTO events(created_at, channel, topic, text, message_id, salience)
         VALUES(?,?,?,?,?,?)`,
		time.Now().Format(time.RFC3339), channel, topic, text, mid, salience,
	)
}

// InsertMemoryItem stores a detail memory with decay parameters.
func InsertMemoryItem(db *sql.DB, channel, topic, key, value string, salience float64, halfLifeDays float64) {
	if db == nil {
		return
	}
	channel = strings.TrimSpace(channel)
	if channel == "" {
		channel = "unknown"
	}
	topic = strings.TrimSpace(topic)
	key = strings.TrimSpace(key)
	value = strings.TrimSpace(value)
	if key == "" || value == "" {
		return
	}
	if salience < 0 {
		salience = 0
	}
	if salience > 1 {
		salience = 1
	}
	if halfLifeDays <= 0 {
		halfLifeDays = 14.0
	}
	_, _ = db.Exec(
		`INSERT INTO memory_items(created_at, channel, topic, key, value, salience, half_life_days)
         VALUES(?,?,?,?,?,?,?)`,
		time.Now().Format(time.RFC3339), channel, topic, key, value, salience, halfLifeDays,
	)
}

// GetLastEpisode returns newest episode summary for active topic (gist).
func GetLastEpisode(db *sql.DB, topic string) (summary string, ok bool) {
	if db == nil || strings.TrimSpace(topic) == "" {
		return "", false
	}
	_ = db.QueryRow(
		`SELECT summary FROM episodes WHERE topic=? ORDER BY id DESC LIMIT 1`,
		topic,
	).Scan(&summary)
	summary = strings.TrimSpace(summary)
	return summary, summary != ""
}

// RecentTurns returns last N (user/reply/auto) messages (chronological).
func RecentTurns(db *sql.DB, n int) string {
	return BuildDialogContext(db, n)
}

type scoredItem struct {
	id    int64
	score float64
	text  string
}

// RecallDetails returns top K memory items by salience * time-decay.
func RecallDetails(db *sql.DB, topic string, k int) string {
	if db == nil || strings.TrimSpace(topic) == "" || k <= 0 {
		return ""
	}
	rows, err := db.Query(
		`SELECT id, created_at, key, value, salience, half_life_days
		 FROM memory_items
		 WHERE topic=?
		 ORDER BY id DESC
		 LIMIT 200`,
		topic,
	)
	if err != nil {
		return ""
	}
	defer rows.Close()
	now := time.Now()
	var items []scoredItem
	for rows.Next() {
		var id int64
		var createdAt, key, value string
		var sal, half float64
		if err := rows.Scan(&id, &createdAt, &key, &value, &sal, &half); err != nil {
			continue
		}
		ts, _ := time.Parse(time.RFC3339, createdAt)
		ageDays := now.Sub(ts).Hours() / 24.0
		if half <= 0 {
			half = 14.0
		}
		decay := math.Pow(0.5, ageDays/half)
		score := clamp01(sal) * decay
		txt := key + ": " + clipForContext(value, 220)
		items = append(items, scoredItem{id: id, score: score, text: txt})
	}
	sort.Slice(items, func(i, j int) bool { return items[i].score > items[j].score })
	if len(items) > k {
		items = items[:k]
	}
	if len(items) == 0 {
		return ""
	}
	var b strings.Builder
	for _, it := range items {
		b.WriteString("- ")
		b.WriteString(it.text)
		b.WriteString("\n")
		_, _ = db.Exec(`UPDATE memory_items SET last_accessed_at=? WHERE id=?`, now.Format(time.RFC3339), it.id)
	}
	return strings.TrimSpace(b.String())
}

// LatencyAffect: pain + sorrow when latency too high.
func LatencyAffect(ws *Workspace, aff *AffectState, eg *epi.Epigenome, latency time.Duration) {
	if ws == nil || aff == nil || eg == nil {
		return
	}
	_, _, _, _, _, painMs, _ := eg.MemoryParams()
	latMs := float64(latency.Milliseconds())
	ws.LastLatencyMs = latMs
	alpha := 0.15
	ws.LatencyEMA = (1-alpha)*ws.LatencyEMA + alpha*latMs

	if latMs >= float64(painMs) {
		aff.Ensure("sorrow", 0.02)
		over := (latMs - float64(painMs)) / float64(painMs)
		if over > 2 {
			over = 2
		}
		aff.Set("pain", clamp01(aff.Get("pain")+0.08*over))
		aff.Set("sorrow", clamp01(aff.Get("sorrow")+0.05*over))
	}
}

// AutoTuneMemory mutates epigenetic memory parameters based on sustained pain/latency.
func AutoTuneMemory(eg *epi.Epigenome, ws *Workspace, aff *AffectState) (mutated bool) {
	if eg == nil || ws == nil || aff == nil {
		return false
	}
	conE, ctxT, _, detHalf, _, painMs, autoTune := eg.MemoryParams()
	if !autoTune {
		return false
	}
	if !ws.lastTuneAt.IsZero() && time.Since(ws.lastTuneAt) < 2*time.Minute {
		return false
	}

	if ws.LatencyEMA > float64(painMs)*1.2 || aff.Get("sorrow") > 0.25 || aff.Get("pain") > 0.35 {
		if ctxT > 6 {
			ctxT -= 2
			mutated = true
		}
		if conE > 8 {
			conE -= 2
			mutated = true
		}
		if detHalf > 7 {
			detHalf -= 2
			mutated = true
		}

		if mutated {
			m := eg.Modules["memory"]
			if m != nil {
				if m.Params == nil {
					m.Params = map[string]any{}
				}
				m.Params["context_turns"] = ctxT
				m.Params["consolidate_every_events"] = conE
				m.Params["detail_half_life_days"] = detHalf
			}
			ws.lastTuneAt = time.Now()
		}
	}
	return mutated
}
