package brain

import (
	"database/sql"
	"strings"
	"time"
)

func UpsertAxiomInterpretation(db *sql.DB, axiomID int, kind, key, value string, confidence float64, sourceNote string) error {
	if db == nil {
		return nil
	}
	kind = strings.TrimSpace(kind)
	key = strings.TrimSpace(key)
	value = strings.TrimSpace(value)
	sourceNote = strings.TrimSpace(sourceNote)
	if axiomID < 1 || axiomID > 4 {
		axiomID = 1
	}
	if kind == "" || key == "" || value == "" {
		return nil
	}
	if confidence < 0 {
		confidence = 0
	}
	if confidence > 1 {
		confidence = 1
	}
	now := time.Now().Format(time.RFC3339)
	_, err := db.Exec(`INSERT INTO axiom_interpretations(axiom_id,kind,key,value,confidence,source_note,updated_at)
		VALUES(?,?,?,?,?,?,?)
		ON CONFLICT(axiom_id,kind,key) DO UPDATE SET value=excluded.value, confidence=excluded.confidence, source_note=excluded.source_note, updated_at=excluded.updated_at`,
		axiomID, kind, key, value, confidence, sourceNote, now)
	return err
}
