package brain

import (
	"database/sql"
	"time"
)

// GetPreference returns a preference value in [-1..1].
// If missing, returns def.
func GetPreference(db *sql.DB, key string, def float64) float64 {
	if db == nil || key == "" {
		return def
	}
	var cur sql.NullFloat64
	_ = db.QueryRow(`SELECT value FROM preferences WHERE key=?`, key).Scan(&cur)
	if !cur.Valid {
		return def
	}
	return clamp11(cur.Float64)
}

// GetPreference01 maps a preference in [-1..1] to [0..1].
func GetPreference01(db *sql.DB, key string, def01 float64) float64 {
	if def01 < 0 {
		def01 = 0
	}
	if def01 > 1 {
		def01 = 1
	}
	v := GetPreference(db, key, 2*def01-1)
	return clamp01((v + 1) / 2)
}

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
