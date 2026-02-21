package brain

import (
	"database/sql"
	"time"
)

func BumpInterest(db *sql.DB, topic string, delta float64) {
	if db == nil {
		return
	}
	topic = normalizeKey(topic)
	if topic == "" {
		return
	}
	// interests(topic PRIMARY KEY, weight REAL, updated_at TEXT)
	// (exists already in your schema)
	_, _ = db.Exec(
		`INSERT INTO interests(topic, weight, updated_at) VALUES(?,?,?)
         ON CONFLICT(topic) DO UPDATE SET
           weight = MAX(0.0, interests.weight + excluded.weight),
           updated_at = excluded.updated_at`,
		topic, delta, time.Now().Format(time.RFC3339),
	)
}

func TopInterest(db *sql.DB) (topic string, weight float64) {
	if db == nil {
		return "", 0
	}
	row := db.QueryRow(`SELECT topic, weight FROM interests ORDER BY weight DESC LIMIT 1`)
	_ = row.Scan(&topic, &weight)
	return topic, weight
}

// Optional gentle decay so interests don't stick forever.
func DecayInterests(db *sql.DB, factor float64) {
	if db == nil {
		return
	}
	if factor <= 0 || factor >= 1 {
		return
	}
	_, _ = db.Exec(`UPDATE interests SET weight = weight * ?`, factor)
}

func normalizeKey(s string) string {
	// keep it super light; your ExtractTopic already reduces.
	if len(s) > 64 {
		return s[:64]
	}
	return s
}
