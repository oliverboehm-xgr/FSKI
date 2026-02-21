package brain

import (
	"database/sql"
	"time"
)

// Daydreaming: background thought generation from Interests + Concepts + Affects + Drives.
// This is kernel-side cognition; no LLM needed.
func TickDaydream(db *sql.DB, ws *Workspace, d *Drives, aff *AffectState, dt time.Duration) {
	if db == nil || ws == nil || d == nil {
		return
	}
	sec := dt.Seconds()
	if sec <= 0 {
		return
	}

	// If curiosity is low or inhibited, daydream less
	if d.Curiosity < 0.25 && d.UrgeToShare < 0.25 {
		return
	}

	topic, w := TopInterest(db)
	if topic == "" || w < 0.05 {
		return
	}

	// Update thought every few seconds, not every tick
	ws._daydreamAccum += sec
	period := 6.0 - 3.0*d.Curiosity // higher curiosity => more frequent thoughts
	if period < 2.0 {
		period = 2.0
	}
	if ws._daydreamAccum < period {
		return
	}
	ws._daydreamAccum = 0

	// Build a short thought using concept summary if available
	c, ok := GetConcept(db, topic)
	content := ""
	if ok && c.Summary != "" {
		content = "Ich kreise um " + topic + ": " + c.Summary
	} else {
		content = "Ich kreise um " + topic + ". Mir fehlt noch eine saubere Einordnung."
	}

	// Salience: interest weight + curiosity - inhibitors
	inhib := 0.0
	if aff != nil {
		inhib = 0.6*aff.Get("shame") + 0.3*aff.Get("fear") + 0.2*aff.Get("pain")
	}
	salience := clamp01(0.3*w + 0.5*d.Curiosity - 0.4*inhib)

	ws.CurrentThought = content
	ws.Confidence = clamp01(0.55 + 0.35*salience)
	ws.LastTopic = topic

	// Mitteilungsbedürfnis wächst, wenn ein Gedanke "salient" ist
	d.UrgeToShare = clamp01(d.UrgeToShare + 0.08*salience)

	// Log thought (memory of internal cognition)
	LogThought(db, "daydream", topic, salience, content)
}

func LogThought(db *sql.DB, kind, topic string, salience float64, content string) {
	if db == nil {
		return
	}
	_, _ = db.Exec(
		`INSERT INTO thought_log(created_at, kind, topic, salience, content)
         VALUES(?,?,?,?,?)`,
		time.Now().Format(time.RFC3339),
		kind, topic, salience, content,
	)
}
