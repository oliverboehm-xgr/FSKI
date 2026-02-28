"""SQLite schema for BunnyCore V1."""

SCHEMA_SQL = r'''
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS axioms(
  axiom_key TEXT PRIMARY KEY, -- e.g. A1..A4
  text TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS state_axes(
  axis_index INTEGER PRIMARY KEY,
  axis_name TEXT UNIQUE NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS state_axes_meta(
  axis_name TEXT PRIMARY KEY,
  invariant INTEGER NOT NULL DEFAULT 0,
  decays INTEGER NOT NULL DEFAULT 0,
  source TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL
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


CREATE TABLE IF NOT EXISTS memory_short(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  role TEXT NOT NULL,
  content TEXT NOT NULL,
  created_at TEXT NOT NULL,
  ui_message_id INTEGER NOT NULL DEFAULT 0,
  topic TEXT NOT NULL DEFAULT '',
  salience REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS memory_long(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  summary TEXT NOT NULL,
  created_at TEXT NOT NULL
);

-- Axiom operationalizations / interpretations (mutable, many-per-axiom)
CREATE TABLE IF NOT EXISTS axiom_interpretations(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  axiom_key TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'rewrite', -- rewrite|definition|metric|rule|example|anti_example
  key TEXT NOT NULL DEFAULT 'latest',
  value TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.4,
  source_note TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL,
  UNIQUE(axiom_key, kind, key)
);


CREATE TABLE IF NOT EXISTS axiom_digests(
  axiom_key TEXT PRIMARY KEY,
  digest TEXT NOT NULL,
  checksum TEXT NOT NULL DEFAULT '',
  updated_at TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS ui_messages(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  kind TEXT NOT NULL, -- user|reply|think|auto
  text TEXT NOT NULL,
  rating INTEGER, -- -1,0,1
  caught INTEGER NOT NULL DEFAULT 0
);


CREATE TABLE IF NOT EXISTS workspace_current(
  id INTEGER PRIMARY KEY CHECK (id=1),
  items_json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  items_json TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS sleep_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  summary_json TEXT NOT NULL
);


CREATE TABLE IF NOT EXISTS adapters(
  event_type TEXT PRIMARY KEY,
  encoder_name TEXT NOT NULL,
  matrix_name TEXT NOT NULL,
  matrix_version INTEGER NOT NULL,
  meta_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS websense_pages(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  query TEXT NOT NULL DEFAULT '',
  url TEXT NOT NULL,
  title TEXT NOT NULL DEFAULT '',
  snippet TEXT NOT NULL DEFAULT '',
  body TEXT NOT NULL DEFAULT '',
  domain TEXT NOT NULL DEFAULT '',
  hash TEXT NOT NULL DEFAULT '',
  ok INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS decision_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '', -- user|idle|daydream
  input_text TEXT NOT NULL DEFAULT '',
  decision_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daydream_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  trigger TEXT NOT NULL DEFAULT '',
  state_json TEXT NOT NULL DEFAULT '{}',
  output_json TEXT NOT NULL
);

-- Learned trust scores for domains / sources used by WebSense (no hardcoded heuristics)
CREATE TABLE IF NOT EXISTS trust_domains(
  domain TEXT PRIMARY KEY,
  trust REAL NOT NULL DEFAULT 0.5, -- 0..1
  n_obs INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);

-- Resource metrics tick log (energy computation inputs)
CREATE TABLE IF NOT EXISTS resources_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  metrics_json TEXT NOT NULL
);

-- Per-organ health metrics (latency/errors) and immutable pain inputs
CREATE TABLE IF NOT EXISTS health_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  organ TEXT NOT NULL,
  ok INTEGER NOT NULL,
  latency_ms REAL NOT NULL,
  error TEXT NOT NULL DEFAULT '',
  metrics_json TEXT NOT NULL DEFAULT '{}'
);

-- Plasticity audit log for matrix updates (used to attribute pain/regressions)
CREATE TABLE IF NOT EXISTS matrix_update_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  event_type TEXT NOT NULL,
  matrix_name TEXT NOT NULL,
  from_version INTEGER NOT NULL,
  to_version INTEGER NOT NULL,
  reward REAL NOT NULL,
  delta_frob REAL NOT NULL,
  pain_before REAL NOT NULL DEFAULT 0.0,
  pain_after REAL NOT NULL DEFAULT 0.0,
  rolled_back INTEGER NOT NULL DEFAULT 0,
  rollback_at TEXT NOT NULL DEFAULT '',
  rollback_notes TEXT NOT NULL DEFAULT '',
  notes TEXT NOT NULL DEFAULT ''
);

-- Organ gating decisions (why an organ did/didn't run)
CREATE TABLE IF NOT EXISTS organ_gate_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  phase TEXT NOT NULL,         -- user|idle
  organ TEXT NOT NULL,
  score REAL NOT NULL,
  threshold REAL NOT NULL,
  want INTEGER NOT NULL,
  data_json TEXT NOT NULL DEFAULT '{}'
);

-- Evidence extraction (claims JSON) derived from WebSense evidence for auditability
CREATE TABLE IF NOT EXISTS evidence_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  query TEXT NOT NULL DEFAULT '',
  question TEXT NOT NULL DEFAULT '',
  evidence_json TEXT NOT NULL
);

-- Parsed user feedback / correction traces (LLM-interpreted). Used for learning/audit.
CREATE TABLE IF NOT EXISTS feedback_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  user_text TEXT NOT NULL,
  last_assistant TEXT NOT NULL,
  parsed_json TEXT NOT NULL
);

-- Structured beliefs / corrections from user feedback (generic knowledge overrides)
CREATE TABLE IF NOT EXISTS beliefs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  subject TEXT NOT NULL,
  predicate TEXT NOT NULL,
  object TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0.7,
  provenance TEXT NOT NULL DEFAULT '',
  topic TEXT NOT NULL DEFAULT '',
  salience REAL NOT NULL DEFAULT 0.0,
  half_life_days REAL NOT NULL DEFAULT 45.0,
  updated_at TEXT NOT NULL DEFAULT ''
);

-- Self-model snapshots (capabilities, organs, costs). Used for self-development proposals.
CREATE TABLE IF NOT EXISTS self_model(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  model_json TEXT NOT NULL
);

-- Mutation proposals (self-development). Human approval required.
CREATE TABLE IF NOT EXISTS mutation_proposals(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  trigger TEXT NOT NULL DEFAULT '',
  proposal_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'proposed',
  user_note TEXT NOT NULL DEFAULT ''
);

-- Current needs and wishes (first-class state objects)
CREATE TABLE IF NOT EXISTS needs_current(
  id INTEGER PRIMARY KEY CHECK (id=1),
  json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS wishes_current(
  id INTEGER PRIMARY KEY CHECK (id=1),
  json TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Topic tracking (to reduce conversation drift)
CREATE TABLE IF NOT EXISTS topics(
  topic TEXT PRIMARY KEY,
  weight REAL NOT NULL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS episodes(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  topic TEXT NOT NULL,
  started_at TEXT NOT NULL,
  ended_at TEXT NOT NULL DEFAULT '',
  summary TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

-- Capability registry and offers (generic embodiment/tooling interface)
CREATE TABLE IF NOT EXISTS capabilities(
  name TEXT PRIMARY KEY,
  kind TEXT NOT NULL DEFAULT 'tool', -- tool|sensor|actor
  meta_json TEXT NOT NULL DEFAULT '{}',
  health_json TEXT NOT NULL DEFAULT '{}',
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS capability_calls(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  name TEXT NOT NULL,
  ok INTEGER NOT NULL,
  latency_ms REAL NOT NULL,
  error TEXT NOT NULL DEFAULT '',
  args_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS resource_offers(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  capability TEXT NOT NULL,
  offer_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'offered' -- offered|accepted|rejected|expired
);

-- Versioning for learned signals (regression-safe)
CREATE TABLE IF NOT EXISTS trust_history(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  domain TEXT NOT NULL,
  prev_trust REAL NOT NULL,
  new_trust REAL NOT NULL,
  delta REAL NOT NULL,
  source TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS beliefs_history(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  belief_id INTEGER NOT NULL,
  op TEXT NOT NULL, -- insert|delete
  row_json TEXT NOT NULL
);



-- Failure clusters derived from repeated errors/caught/selfeval (for curriculum + development)
CREATE TABLE IF NOT EXISTS failure_clusters(
  cluster_key TEXT PRIMARY KEY,
  label TEXT NOT NULL DEFAULT '',
  embedding_json TEXT NOT NULL DEFAULT '[]',
  count INTEGER NOT NULL DEFAULT 0,
  last_seen TEXT NOT NULL,
  examples_json TEXT NOT NULL DEFAULT '[]',
  stats_json TEXT NOT NULL DEFAULT '{}'
);

-- Learned skills/strategies to address recurring failure clusters (compact, testable)
CREATE TABLE IF NOT EXISTS skills(
  skill_key TEXT PRIMARY KEY,
  cluster_key TEXT NOT NULL,
  strategy_digest TEXT NOT NULL,
  tests_json TEXT NOT NULL DEFAULT '[]',
  confidence REAL NOT NULL DEFAULT 0.4,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(cluster_key) REFERENCES failure_clusters(cluster_key)
);

-- DevLab generated patches and test reports (audit trail)
CREATE TABLE IF NOT EXISTS devlab_runs(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  cluster_key TEXT NOT NULL DEFAULT '',
  intent TEXT NOT NULL DEFAULT '',
  patch_diff TEXT NOT NULL DEFAULT '',
  test_plan_json TEXT NOT NULL DEFAULT '[]',
  test_output TEXT NOT NULL DEFAULT '',
  verdict TEXT NOT NULL DEFAULT '',
  meta_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS sensory_tokens(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  modality TEXT NOT NULL,          -- vision|audio
  summary TEXT NOT NULL,           -- compressed semantic summary
  tokens_json TEXT NOT NULL,       -- structured attributes
  salience REAL NOT NULL DEFAULT 0.0,
  topic TEXT NOT NULL DEFAULT '',
  created_at TEXT NOT NULL
);

'''
