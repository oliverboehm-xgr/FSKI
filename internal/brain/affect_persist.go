package brain

import (
	"database/sql"
	"time"
)

func LoadAffectState(db *sql.DB, a *AffectState) error {
	if db == nil || a == nil {
		return nil
	}
	rows, err := db.Query(`SELECT name, value FROM affect_state`)
	if err != nil {
		return err
	}
	defer rows.Close()
	for rows.Next() {
		var name string
		var v float64
		if err := rows.Scan(&name, &v); err != nil {
			continue
		}
		a.Set(name, v)
	}
	return nil
}

func SaveAffectState(db *sql.DB, a *AffectState) error {
	if db == nil || a == nil {
		return nil
	}
	now := time.Now().Format(time.RFC3339)
	for _, k := range a.Keys() {
		v := a.Get(k)
		_, _ = db.Exec(
			`INSERT INTO affect_state(name,value,updated_at) VALUES(?,?,?)
             ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
			k, v, now,
		)
	}
	return nil
}
