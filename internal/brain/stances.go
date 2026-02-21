package brain

import (
	"database/sql"
	"math"
	"strings"
	"time"
)

type Stance struct {
	Topic        string
	Position     float64
	Label        string
	Rationale    string
	Confidence   float64
	UpdatedAt    time.Time
	HalfLifeDays float64
}

func GetStance(db *sql.DB, topic string) (Stance, bool) {
	if db == nil || strings.TrimSpace(topic) == "" {
		return Stance{}, false
	}
	var s Stance
	var ts string
	err := db.QueryRow(`SELECT topic, position, label, rationale, confidence, updated_at, half_life_days FROM stances WHERE topic=?`, topic).
		Scan(&s.Topic, &s.Position, &s.Label, &s.Rationale, &s.Confidence, &ts, &s.HalfLifeDays)
	if err != nil || s.Topic == "" {
		return Stance{}, false
	}
	t, _ := time.Parse(time.RFC3339, ts)
	s.UpdatedAt = t
	return s, true
}

func SaveStance(db *sql.DB, s Stance) {
	if db == nil || strings.TrimSpace(s.Topic) == "" {
		return
	}
	if s.Position < -1 {
		s.Position = -1
	}
	if s.Position > 1 {
		s.Position = 1
	}
	s.Confidence = clamp01(s.Confidence)
	if s.HalfLifeDays <= 0 {
		s.HalfLifeDays = 60
	}
	if strings.TrimSpace(s.Label) == "" {
		s.Label = "neutral"
	}
	if strings.TrimSpace(s.Rationale) == "" {
		s.Rationale = "-"
	}
	ts := time.Now().Format(time.RFC3339)
	_, _ = db.Exec(`INSERT INTO stances(topic, position, label, rationale, confidence, updated_at, half_life_days)
		VALUES(?,?,?,?,?,?,?)
		ON CONFLICT(topic) DO UPDATE SET
		position=excluded.position,
		label=excluded.label,
		rationale=excluded.rationale,
		confidence=excluded.confidence,
		updated_at=excluded.updated_at,
		half_life_days=excluded.half_life_days`,
		s.Topic, s.Position, s.Label, s.Rationale, s.Confidence, ts, s.HalfLifeDays)
}

func AddStanceSource(db *sql.DB, topic, url, domain, snippet, fetchedAt string) {
	if db == nil || topic == "" || url == "" {
		return
	}
	_, _ = db.Exec(`INSERT OR IGNORE INTO stance_sources(topic,url,domain,snippet,fetched_at) VALUES(?,?,?,?,?)`,
		topic, url, domain, snippet, fetchedAt)
}

func StanceConfidenceDecayed(s Stance) float64 {
	if s.UpdatedAt.IsZero() {
		return s.Confidence
	}
	ageDays := time.Since(s.UpdatedAt).Hours() / 24.0
	half := s.HalfLifeDays
	if half <= 0 {
		half = 60
	}
	decay := math.Pow(0.5, ageDays/half)
	return clamp01(s.Confidence * decay)
}
