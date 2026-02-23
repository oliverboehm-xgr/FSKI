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

		// Unified event stream (multi-channel: user/reply/auto/daydream/web/...)
		`CREATE TABLE IF NOT EXISTS events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			channel TEXT NOT NULL,
			topic TEXT NOT NULL,
			text TEXT NOT NULL,
			message_id INTEGER,
			salience REAL NOT NULL DEFAULT 0.3
		);`,
		`CREATE INDEX IF NOT EXISTS idx_events_created_at ON events(created_at);`,
		`CREATE INDEX IF NOT EXISTS idx_events_topic ON events(topic);`,

		// Episodes (gist/story). Details fade, gist remains.
		`CREATE TABLE IF NOT EXISTS episodes (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			topic TEXT NOT NULL,
			start_event_id INTEGER NOT NULL,
			end_event_id INTEGER NOT NULL,
			summary TEXT NOT NULL,
			salience REAL NOT NULL DEFAULT 0.6
		);`,
		`CREATE INDEX IF NOT EXISTS idx_episodes_topic ON episodes(topic);`,

		// Memory items (details with decay)
		`CREATE TABLE IF NOT EXISTS memory_items (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			channel TEXT NOT NULL,
			topic TEXT NOT NULL,
			key TEXT NOT NULL,
			value TEXT NOT NULL,
			salience REAL NOT NULL DEFAULT 0.3,
			half_life_days REAL NOT NULL DEFAULT 14.0,
			last_accessed_at TEXT
		);`,
		`CREATE INDEX IF NOT EXISTS idx_memory_items_topic ON memory_items(topic);`,

		// Values & stances
		`CREATE TABLE IF NOT EXISTS stances (
			topic TEXT PRIMARY KEY,
			position REAL NOT NULL,
			label TEXT NOT NULL,
			rationale TEXT NOT NULL,
			confidence REAL NOT NULL,
			updated_at TEXT NOT NULL,
			half_life_days REAL NOT NULL DEFAULT 60.0
		);`,
		`CREATE TABLE IF NOT EXISTS stance_sources (
			topic TEXT NOT NULL,
			url TEXT NOT NULL,
			domain TEXT NOT NULL,
			snippet TEXT NOT NULL,
			fetched_at TEXT NOT NULL,
			PRIMARY KEY(topic, url)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_stance_sources_topic ON stance_sources(topic);`,

		// Generic key/value state (throttles, counters)
		`CREATE TABLE IF NOT EXISTS kv_state (
			key TEXT PRIMARY KEY,
			value TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);`,

		// Generic semantic long-term memory (facts)
		`CREATE TABLE IF NOT EXISTS facts (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			subject TEXT NOT NULL,
			predicate TEXT NOT NULL,
			object TEXT NOT NULL,
			confidence REAL NOT NULL,
			salience REAL NOT NULL,
			half_life_days REAL NOT NULL,
			source TEXT NOT NULL,
			created_at TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			UNIQUE(subject, predicate)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);`,
		`CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);`,

		// Schema proposals (generic table-evolution proposals)
		`CREATE TABLE IF NOT EXISTS schema_proposals (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			title TEXT NOT NULL,
			sql TEXT NOT NULL,
			status TEXT NOT NULL,
			notes TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_schema_proposals_status ON schema_proposals(status);`,

		// ---------- ResourceSpace ----------
		`CREATE TABLE IF NOT EXISTS resources (
			id TEXT PRIMARY KEY,              -- e.g. disk:C:, ram, cpu
			kind TEXT NOT NULL,               -- capacity|sensor|capability
			present INTEGER NOT NULL,          -- 0/1
			metrics_json TEXT NOT NULL,        -- JSON
			constraints_json TEXT NOT NULL,    -- JSON
			updated_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_resources_kind ON resources(kind);`,

		`CREATE TABLE IF NOT EXISTS expand_candidates (
			id TEXT PRIMARY KEY,              -- e.g. expand:ram:upgrade
			yields_json TEXT NOT NULL,         -- JSON list of resource ids/capabilities
			prereq_json TEXT NOT NULL,         -- JSON list
			cost REAL NOT NULL,                -- 0..1
			evidence REAL NOT NULL,            -- 0..1
			helps_json TEXT NOT NULL,          -- JSON map need->strength
			updated_at TEXT NOT NULL
		);`,

		`CREATE TABLE IF NOT EXISTS candidate_history (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			candidate_id TEXT NOT NULL,
			outcome TEXT NOT NULL,             -- proposed|accepted|rejected|succeeded|failed
			note TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_candidate_history_candidate ON candidate_history(candidate_id);`,

		// ---------- Caught events (for user satisfaction / shame learning) ----------
		`CREATE TABLE IF NOT EXISTS caught_events (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			message_id INTEGER NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_caught_events_created_at ON caught_events(created_at);`,

		// ---------- Code index (self-awareness) ----------
		`CREATE TABLE IF NOT EXISTS code_index (
			path TEXT PRIMARY KEY,
			package TEXT NOT NULL,
			summary TEXT NOT NULL,
			symbols_json TEXT NOT NULL,
			updated_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_code_index_package ON code_index(package);`,

		// ---------- Code proposals (gated self-modifying code) ----------
		`CREATE TABLE IF NOT EXISTS code_proposals (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			title TEXT NOT NULL,
			diff TEXT NOT NULL,
			status TEXT NOT NULL, -- proposed|applied|rejected
			notes TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_code_proposals_status ON code_proposals(status);`,

		// ---------- Epigenome proposals (gated self-modifying config) ----------
		`CREATE TABLE IF NOT EXISTS epigenome_proposals (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			title TEXT NOT NULL,
			patch_json TEXT NOT NULL,
			status TEXT NOT NULL, -- proposed|applied|rejected
			notes TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_epigenome_proposals_status ON epigenome_proposals(status);`,

		// ---------- Axiom system (kernel axioms + learned interpretations) ----------
		`CREATE TABLE IF NOT EXISTS axiom_interpretations (
			axiom_id INTEGER NOT NULL,
			kind TEXT NOT NULL,             -- definition|example|anti_example|metric|rule
			key TEXT NOT NULL,              -- e.g. harm:financial, serve:task_completion
			value TEXT NOT NULL,            -- JSON or text
			confidence REAL NOT NULL,
			source_note TEXT NOT NULL,
			updated_at TEXT NOT NULL,
			PRIMARY KEY(axiom_id, kind, key)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_axiom_interpretations_axiom ON axiom_interpretations(axiom_id);`,

		// Every autonomous self-change is logged (for transparency + rollback).
		`CREATE TABLE IF NOT EXISTS self_changes (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			kind TEXT NOT NULL,             -- concept|axiom|epigenome|lora|code|policy
			target TEXT NOT NULL,           -- e.g. axiom:2 harm_def, epi:autonomy.cooldown
			delta_json TEXT NOT NULL,       -- JSON merge patch or change payload
			axiom_goal INTEGER NOT NULL,    -- 1..4 (what it tries to improve)
			allowed INTEGER NOT NULL,       -- 0/1
			axiom_block INTEGER NOT NULL,   -- 0 if allowed, else 1..4
			risk TEXT NOT NULL,             -- low|med|high|unknown
			energy_cost REAL NOT NULL,
			note TEXT NOT NULL,
			rollback_key TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_self_changes_created_at ON self_changes(created_at);`,
		`CREATE INDEX IF NOT EXISTS idx_self_changes_kind ON self_changes(kind);`,

		// ---------- A/B trials (preference data for LoRA / behavior) ----------
		`CREATE TABLE IF NOT EXISTS ab_trials (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			prompt TEXT NOT NULL,
			a_model TEXT NOT NULL,
			a_text TEXT NOT NULL,
			b_model TEXT NOT NULL,
			b_text TEXT NOT NULL,
			status TEXT NOT NULL, -- open|chosen
			choice TEXT NOT NULL,
			chosen_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_ab_trials_status ON ab_trials(status);`,

		// ---------- Intent classifier (Naive Bayes) ----------
		`CREATE TABLE IF NOT EXISTS reply_context (
			message_id INTEGER PRIMARY KEY,
			user_text TEXT NOT NULL,
			intent TEXT NOT NULL,
			created_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_reply_context_intent ON reply_context(intent);`,

		`CREATE TABLE IF NOT EXISTS intent_nb_prior (
			intent TEXT PRIMARY KEY,
			count REAL NOT NULL
		);`,

		`CREATE TABLE IF NOT EXISTS intent_nb_token (
			token TEXT NOT NULL,
			intent TEXT NOT NULL,
			count REAL NOT NULL,
			PRIMARY KEY(token, intent)
		);`,
		`CREATE TABLE IF NOT EXISTS intent_nb_meta (
			intent TEXT PRIMARY KEY,
			token_total REAL NOT NULL
		);`,

		// ---------- Preferences (likes/dislikes) ----------
		`CREATE TABLE IF NOT EXISTS preferences (
			key TEXT PRIMARY KEY,
			value REAL NOT NULL,
			updated_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_preferences_key ON preferences(key);`,

		// ---------- Policy Bandit (Thompson sampling) ----------
		`CREATE TABLE IF NOT EXISTS policy_stats (
			context_key TEXT NOT NULL,
			action TEXT NOT NULL,
			alpha REAL NOT NULL,
			beta REAL NOT NULL,
			updated_at TEXT NOT NULL,
			PRIMARY KEY(context_key, action)
		);`,
		`CREATE INDEX IF NOT EXISTS idx_policy_stats_ctx ON policy_stats(context_key);`,

		// Extended reply context with policy data.
		`CREATE TABLE IF NOT EXISTS reply_context_v2 (
			message_id INTEGER PRIMARY KEY,
			user_text TEXT NOT NULL,
			intent TEXT NOT NULL,
			policy_ctx TEXT NOT NULL,
			action TEXT NOT NULL,
			style TEXT NOT NULL,
			created_at TEXT NOT NULL
		);`,

		// ---------- Token stats for generic informativeness gate ----------
		// Document frequency of tokens across user utterances. Used for learned IDF.
		`CREATE TABLE IF NOT EXISTS token_df (
			token TEXT PRIMARY KEY,
			df INTEGER NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_token_df_df ON token_df(df);`,

		// A/B training trials (split chat, choose A or B)
		`CREATE TABLE IF NOT EXISTS train_trials (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			user_msg_id INTEGER NOT NULL,
			topic TEXT NOT NULL,
			intent TEXT NOT NULL,
			ctx_key TEXT NOT NULL,
			a_action TEXT NOT NULL,
			a_style TEXT NOT NULL,
			a_text TEXT NOT NULL,
			b_action TEXT NOT NULL,
			b_style TEXT NOT NULL,
			b_text TEXT NOT NULL,
			chosen TEXT NOT NULL,
			note TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_train_trials_created ON train_trials(created_at);`,

		// Pending thought proposals queue (optional; lets Bunny propose asynchronously)
		`CREATE TABLE IF NOT EXISTS thought_proposals (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			kind TEXT NOT NULL,
			title TEXT NOT NULL,
			payload TEXT NOT NULL,
			status TEXT NOT NULL,
			note TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_thought_proposals_status ON thought_proposals(status);`,

		// ---------- Evolution tournament (daily epigenome forks) ----------
		`CREATE TABLE IF NOT EXISTS evolution_runs (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			created_at TEXT NOT NULL,
			window_start TEXT NOT NULL,
			window_end TEXT NOT NULL,
			fork_count INTEGER NOT NULL,
			budget_seconds INTEGER NOT NULL,
			weights_json TEXT NOT NULL,
			winner_index INTEGER NOT NULL,
			winner_score REAL NOT NULL,
			notes TEXT NOT NULL
		);`,
		`CREATE TABLE IF NOT EXISTS evolution_candidates (
			id INTEGER PRIMARY KEY AUTOINCREMENT,
			run_id INTEGER NOT NULL,
			candidate_index INTEGER NOT NULL,
			title TEXT NOT NULL,
			patch_json TEXT NOT NULL,
			user_reward REAL NOT NULL,
			evidence REAL NOT NULL,
			cost REAL NOT NULL,
			spam REAL NOT NULL,
			coherence REAL NOT NULL,
			fitness REAL NOT NULL,
			created_at TEXT NOT NULL
		);`,
		`CREATE INDEX IF NOT EXISTS idx_evolution_candidates_run ON evolution_candidates(run_id);`,

		`CREATE INDEX IF NOT EXISTS idx_ratings_message_id ON ratings(message_id);`,
	}
	for _, s := range stmts {
		if _, err := db.Exec(s); err != nil {
			return err
		}
	}
	return nil
}
