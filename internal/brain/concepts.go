package brain

import (
	"database/sql"
	"time"
)

type Concept struct {
	Term       string
	Kind       string
	Summary    string
	Confidence float64
	Importance float64
}

func ConceptExists(db *sql.DB, term string) bool {
	if db == nil {
		return false
	}
	var t string
	_ = db.QueryRow(`SELECT term FROM concepts WHERE term=?`, term).Scan(&t)
	return t != ""
}

func GetConcept(db *sql.DB, term string) (Concept, bool) {
	if db == nil {
		return Concept{}, false
	}
	var c Concept
	err := db.QueryRow(
		`SELECT term, kind, summary, confidence, importance FROM concepts WHERE term=?`,
		term,
	).Scan(&c.Term, &c.Kind, &c.Summary, &c.Confidence, &c.Importance)
	if err != nil || c.Term == "" {
		return Concept{}, false
	}
	return c, true
}

func UpsertConcept(db *sql.DB, c Concept) {
	if db == nil {
		return
	}
	now := time.Now().Format(time.RFC3339)
	if c.Kind == "" {
		c.Kind = "unknown"
	}
	if c.Confidence < 0 {
		c.Confidence = 0
	}
	if c.Confidence > 1 {
		c.Confidence = 1
	}
	if c.Importance < 0 {
		c.Importance = 0
	}
	if c.Importance > 1 {
		c.Importance = 1
	}
	_, _ = db.Exec(
		`INSERT INTO concepts(term, kind, summary, confidence, importance, updated_at)
         VALUES(?,?,?,?,?,?)
         ON CONFLICT(term) DO UPDATE SET
           kind=excluded.kind,
           summary=excluded.summary,
           confidence=excluded.confidence,
           importance=excluded.importance,
           updated_at=excluded.updated_at`,
		c.Term, c.Kind, c.Summary, c.Confidence, c.Importance, now,
	)
}

func AddConceptSource(db *sql.DB, term, url, domain, snippet, fetchedAt string) {
	if db == nil || term == "" || url == "" {
		return
	}
	_, _ = db.Exec(
		`INSERT OR IGNORE INTO concept_sources(term,url,domain,snippet,fetched_at) VALUES(?,?,?,?,?)`,
		term, url, domain, snippet, fetchedAt,
	)
}
