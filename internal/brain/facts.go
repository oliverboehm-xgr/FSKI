package brain

import (
	"database/sql"
	"strings"
	"time"
)

type Fact struct {
	Subject      string
	Predicate    string
	Object       string
	Confidence   float64
	Salience     float64
	HalfLifeDays float64
	Source       string
}

func UpsertFact(db *sql.DB, f Fact) {
	if db == nil {
		return
	}
	f.Subject = strings.TrimSpace(f.Subject)
	f.Predicate = strings.TrimSpace(f.Predicate)
	f.Object = strings.TrimSpace(f.Object)
	if f.Subject == "" || f.Predicate == "" || f.Object == "" {
		return
	}
	if f.Confidence <= 0 {
		f.Confidence = 0.7
	}
	if f.Salience <= 0 {
		f.Salience = 0.5
	}
	if f.HalfLifeDays <= 0 {
		f.HalfLifeDays = 365
	}
	if f.Source == "" {
		f.Source = "user"
	}
	now := time.Now().Format(time.RFC3339)
	_, _ = db.Exec(`INSERT INTO facts(subject,predicate,object,confidence,salience,half_life_days,source,created_at,updated_at)
		 VALUES(?,?,?,?,?,?,?,?,?)
		 ON CONFLICT(subject,predicate) DO UPDATE SET object=excluded.object, confidence=excluded.confidence, salience=excluded.salience, half_life_days=excluded.half_life_days, source=excluded.source, updated_at=excluded.updated_at`,
		f.Subject, f.Predicate, f.Object, f.Confidence, f.Salience, f.HalfLifeDays, f.Source, now, now,
	)
}

func GetFact(db *sql.DB, subject, predicate string) (string, bool) {
	if db == nil {
		return "", false
	}
	subject = strings.TrimSpace(subject)
	predicate = strings.TrimSpace(predicate)
	if subject == "" || predicate == "" {
		return "", false
	}
	var obj string
	_ = db.QueryRow(`SELECT object FROM facts WHERE subject=? AND predicate=?`, subject, predicate).Scan(&obj)
	obj = strings.TrimSpace(obj)
	return obj, obj != ""
}
