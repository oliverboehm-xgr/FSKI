"""SQLite schema for BunnyCore V1."""

SCHEMA_SQL = r'''
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_axes(
  axis_index INTEGER PRIMARY KEY,
  axis_name TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_current(
  id INTEGER PRIMARY KEY CHECK (id=1),
  vec_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_snapshots(
  snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
  vec_json TEXT NOT NULL,
  why_json TEXT NOT NULL DEFAULT '[]',
  tags_json TEXT NOT NULL DEFAULT '[]',
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_log(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_type TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS matrices(
  name TEXT NOT NULL,
  version INTEGER NOT NULL,
  op_type TEXT NOT NULL,
  n_rows INTEGER NOT NULL,
  n_cols INTEGER NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  created_at TEXT NOT NULL,
  parent_version INTEGER,
  PRIMARY KEY(name, version)
);

CREATE TABLE IF NOT EXISTS matrix_entries(
  name TEXT NOT NULL,
  version INTEGER NOT NULL,
  i INTEGER NOT NULL,
  j INTEGER NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY(name, version, i, j),
  FOREIGN KEY(name, version) REFERENCES matrices(name, version)
);

CREATE TABLE IF NOT EXISTS adapters(
  event_type TEXT PRIMARY KEY,
  encoder_name TEXT NOT NULL,
  matrix_name TEXT NOT NULL,
  matrix_version INTEGER NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL
);
'''
