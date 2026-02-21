package state

import (
	"database/sql"

	_ "github.com/mattn/go-sqlite3"
)

type DB struct{ *sql.DB }

func Open(path string) (*DB, error) {
	db, err := sql.Open("sqlite3", path)
	if err != nil {
		return nil, err
	}
	if err := migrate(db); err != nil {
		_ = db.Close()
		return nil, err
	}
	return &DB{DB: db}, nil
}

func migrate(db *sql.DB) error {
	stmts := []string{
		`PRAGMA journal_mode=WAL;`,
		`CREATE TABLE IF NOT EXISTS sources (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			url TEXT NOT NULL,
			domain TEXT NOT NULL,
			title TEXT,
			fetched_at TEXT NOT NULL,
			content_hash TEXT NOT NULL,
			snippet TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);`,
		`CREATE TABLE IF NOT EXISTS interests (
			topic TEXT PRIMARY KEY,
			weight REAL NOT NULL,
			updated_at TEXT NOT NULL
		);`,
		`CREATE TABLE IF NOT EXISTS messages (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			priority REAL NOT NULL,
			text TEXT NOT NULL,
			sources_json TEXT NOT NULL
		);`,
		`CREATE TABLE IF NOT EXISTS ratings (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			message_id INTEGER NOT NULL,
			value INTEGER NOT NULL
		);`,
		`CREATE TABLE IF NOT EXISTS traits (
			key TEXT PRIMARY KEY,
			value REAL NOT NULL,
			updated_at TEXT NOT NULL
		);`,

		// Persisted affect state (values 0..1)
		`CREATE TABLE IF NOT EXISTS affect_state (
			name TEXT PRIMARY KEY,
			value REAL NOT NULL,
			updated_at TEXT NOT NULL
		);`,

		// Generic concept store (for any topic, including affect candidates)
		`CREATE TABLE IF NOT EXISTS concepts (
			term TEXT PRIMARY KEY,
			kind TEXT NOT NULL,
			summary TEXT NOT NULL,
			confidence REAL NOT NULL,
			importance REAL NOT NULL,
			updated_at TEXT NOT NULL
		);`,
		`CREATE TABLE IF NOT EXISTS concept_sources (
			term TEXT NOT NULL,
			url TEXT NOT NULL,
			domain TEXT NOT NULL,
			snippet TEXT NOT NULL,
			fetched_at TEXT NOT NULL,
			PRIMARY KEY(term, url)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_concepts_kind ON concepts(kind);`,
		`CREATE INDEX IF NOT EXISTS idx_concept_sources_term ON concept_sources(term);`,

		// Persisted drives (curiosity, urge_to_share, etc.)
		`CREATE TABLE IF NOT EXISTS drive_state (
			key TEXT PRIMARY KEY,
			value REAL NOT NULL,
			updated_at TEXT NOT NULL
		);`,

		// Thought log (tagträumen / internal thoughts)
		`CREATE TABLE IF NOT EXISTS thought_log (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			kind TEXT NOT NULL,
			topic TEXT NOT NULL,
			salience REAL NOT NULL,
			content TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_thought_log_topic ON thought_log(topic);`,

		// message metadata for UI (kind: auto|reply|think)
		`CREATE TABLE IF NOT EXISTS message_meta (
			message_id INTEGER PRIMARY KEY,
			kind TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_message_meta_kind ON message_meta(kind);`,

		// Thread / dialog state (short-term memory anchor)
		`CREATE TABLE IF NOT EXISTS thread_state (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);`,

		`CREATE INDEX IF NOT EXISTS idx_ratings_message_id ON ratings(message_id);`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}
	return nil
}
