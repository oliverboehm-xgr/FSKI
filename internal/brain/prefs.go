package brain

import (
	"database/sql"
	"time"
)

func clamp11(x float64) float64 {
	if x < -1 {
		return -1
	}
	if x > 1 {
		return 1
	}
	return x
}

// UpdatePreferenceEMA updates a preference key in [-1..1] using EMA.
func UpdatePreferenceEMA(db *sql.DB, key string, reward float64, alpha float64) {
	if db == nil || key == "" {
		return
	}
	if alpha <= 0 || alpha > 1 {
		alpha = 0.15
	}
	reward = clamp11(reward)

	var cur float64
	_ = db.QueryRow(`SELECT value FROM preferences WHERE key=?`, key).Scan(&cur)
	next := (1-alpha)*cur + alpha*reward
	next = clamp11(next)

	_, _ = db.Exec(
		`INSERT INTO preferences(key,value,updated_at) VALUES(?,?,?)
		 ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		key, next, time.Now().Format(time.RFC3339),
	)
}
