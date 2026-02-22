package epi

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"
)

type ModuleSpec struct {
	Type    string         `json:"type"`
	Enabled bool           `json:"enabled"`
	Params  map[string]any `json:"params,omitempty"`
}

type AffectDef struct {
	Baseline       float64 `json:"baseline"`
	DecayPerSec    float64 `json:"decayPerSec"`
	EnergyCoupling float64 `json:"energyCoupling"`
}

type Epigenome struct {
	Version       int                    `json:"version"`
	Modules       map[string]*ModuleSpec `json:"modules"`
	AffectDefsMap map[string]AffectDef   `json:"affect_defs,omitempty"`
}

func LoadOrInit(path string) (*Epigenome, error) {
	b, err := os.ReadFile(path)
	if err == nil {
		var eg Epigenome
		if err := json.Unmarshal(b, &eg); err != nil {
			return nil, err
		}
		if eg.Modules == nil {
			eg.Modules = map[string]*ModuleSpec{}
		}
		if eg.AffectDefsMap == nil {
			eg.AffectDefsMap = map[string]AffectDef{}
		}
		changed := eg.ensureDefaults()
		if changed {
			if err := eg.Save(path); err != nil {
				fmt.Println("WARN: failed to persist epigenome migration:", err)
			}
		}
		return &eg, nil
	}
	if !os.IsNotExist(err) {
		return nil, err
	}

	eg := &Epigenome{
		Version: 1,
		Modules: map[string]*ModuleSpec{
			"locale":           {Type: "locale", Enabled: true, Params: map[string]any{"lang": "de"}},
			"energy":           {Type: "energy", Enabled: true, Params: map[string]any{"max": 100}},
			"utterance_filter": {Type: "utterance_filter", Enabled: true, Params: map[string]any{"banned_phrases": []any{}}},
			"heartbeat":        {Type: "heartbeat", Enabled: true, Params: map[string]any{"ms": 500}},
			"auto_speak":       {Type: "auto_speak", Enabled: true, Params: map[string]any{"cooldown_seconds": 18}},
			"memory": {Type: "memory", Enabled: true, Params: map[string]any{
				"consolidate_every_events": 16,
				"context_turns":            10,
				"detail_items":             6,
				"detail_half_life_days":    14.0,
				"episode_half_life_days":   120.0,
				"latency_pain_ms":          2500,
				"auto_tune":                true,
			}},
			"values": {Type: "values", Enabled: true, Params: map[string]any{
				"minimize_suffering": 1.0,
				"truthfulness":       1.0,
				"fairness":           0.8,
				"stability":          0.6,
				"curiosity":          0.6,
			}},
			"stance": {Type: "stance", Enabled: true, Params: map[string]any{
				"half_life_days": 60.0,
				"min_confidence": 0.35,
				"auto_update":    true,
			}},
			"scout": {Type: "scout", Enabled: true, Params: map[string]any{
				"interval_seconds": 45,
				"min_curiosity":    0.55,
				"max_per_hour":     24,
			}},

			// Cortex bus + human-like daydreaming (images + inner speech)
			"cortex_bus": {Type: "cortex_bus", Enabled: true, Params: map[string]any{
				"tick_ms": 500,
			}},
			"daydream": {Type: "daydream", Enabled: true, Params: map[string]any{
				"interval_seconds": 20,
				"min_curiosity":    0.45,
				"min_energy":       8,
				"visual_weight":    0.55, // how visual vs verbal the thought is (0..1)
			}},
			"critic": {Type: "critic", Enabled: true, Params: map[string]any{
				"enabled":       true,
				"max_sentences": 8,
			}},

			// Drives v1: survival (resources), curiosity (knowledge gap), user satisfaction (ratings), social (decay)
			"drives_v1": {Type: "drives_v1", Enabled: true, Params: map[string]any{
				"disk_path":         "C:\\",
				"disk_target_bytes": 10000000000, // 10GB
				"ram_target_bytes":  3000000000,  // 3GB
				"latency_target_ms": 2500,

				"w_disk": 0.30,
				"w_ram":  0.30,
				"w_cpu":  0.20,
				"w_lat":  0.15,
				"w_err":  0.05,

				"k_disk": 4.0,
				"k_ram":  5.0,
				"k_cpu":  3.0,

				"tau_social_seconds":        1200, // 20min
				"ema_user_reward":           0.12,
				"ema_caught":                0.20,
				"help_min_interval_seconds": 180, // don't nag
			}},

			// Models per brain area (LoRA-ready).
			// Keys are "speaker", "critic", "daydream", "scout", "hippocampus", "stance".
			"models": {Type: "models", Enabled: true, Params: map[string]any{
				// Default split: heavy model for creative/complex synthesis,
				// small model for checking/extracting.
				"default":  "llama3.1:8b",
				"speaker":  "llama3.1:8b",
				"daydream": "llama3.1:8b",
				"stance":   "llama3.1:8b",

				// Small / efficient
				"critic":      "llama3.2:3b",
				"scout":       "llama3.2:3b",
				"hippocampus": "llama3.2:3b",
				// later you can set: "critic": "llama3.1:8b-lora-critic"
			}},

			// Online intent classifier (Naive Bayes) parameters
			"intent_nb": {Type: "intent_nb", Enabled: true, Params: map[string]any{
				"enabled":    true,
				"min_tokens": 2,
				"threshold":  0.72, // only trust if P(best) >= threshold
				"alpha":      1.0,  // Laplace smoothing
			}},

			// Generic "informativeness gate": blocks research/stance/topic drift/training for low-info utterances.
			// Learns IDF from token_df + global doc count stored in kv_state.
			"info_gate": {Type: "info_gate", Enabled: true, Params: map[string]any{
				"enabled":             true,
				"min_info":            0.33, // below => treat as low-info (smalltalk/ack)
				"idf_threshold":       1.0,  // baseline for booster
				"idf_2char_threshold": 3.0,  // allow rare 2-char tokens (AI, VW) if truly informative
				"stopword_ratio":      0.35, // df/N above => treated as non-informative
				"min_tokens":          1,    // gate considers even 1-word utterances
				"warmup_min_docs":     30,   // learned stopword filter starts only after enough docs
				"stopword_min_df":     8,    // token must be frequently observed before suppression
			}},

			// Generic semantic long-term memory (facts)
			"semantic_memory": {Type: "semantic_memory", Enabled: true, Params: map[string]any{
				"enabled":             true,
				"max_writes_per_turn": 3,
				"max_reads_per_turn":  2,
				"write_rules": []any{
					map[string]any{
						"name":           "user_name_de_1",
						"regex":          `(?i)\bich\s+hei(?:ß|ss)e\s+([A-Za-zÄÖÜäöüß\-]{2,32})\b`,
						"subject":        "user",
						"predicate":      "name",
						"object":         "$1",
						"confidence":     0.95,
						"salience":       0.80,
						"half_life_days": 3650,
						"source":         "user",
						"ack":            "Okay. Ich speichere das im Langzeitgedächtnis: Dein Name ist {{object}}.",
					},
					map[string]any{
						"name":           "user_name_de_2",
						"regex":          `(?i)\bmein\s+name\s+ist\s+([A-Za-zÄÖÜäöüß\-]{2,32})\b`,
						"subject":        "user",
						"predicate":      "name",
						"object":         "$1",
						"confidence":     0.95,
						"salience":       0.80,
						"half_life_days": 3650,
						"source":         "user",
						"ack":            "Okay. Ich speichere das im Langzeitgedächtnis: Dein Name ist {{object}}.",
					},
					map[string]any{
						"name":           "remember_request_de",
						"regex":          `(?i)\b(merk|speicher)\w*\s+dir\s+(.{3,80})$`,
						"subject":        "user",
						"predicate":      "remember_request",
						"object":         "$2",
						"confidence":     0.70,
						"salience":       0.50,
						"half_life_days": 365,
						"source":         "user",
						"ack":            "Okay. Ich speichere das im Langzeitgedächtnis: {{object}}.",
					},
				},
				"read_rules": []any{
					map[string]any{
						"name":           "ask_user_name_de",
						"regex":          `(?i)\b(kennst|wei(?:ß|ss)t)\s+du\s+meinen\s+namen\b|\bmeinen\s+namen\?\b|\bmeine[n]?\s+name[n]?\b`,
						"subject":        "user",
						"predicate":      "name",
						"answer_found":   "Ja. Du heißt {{object}}.",
						"answer_missing": "Noch nicht. Wie heißt du?",
					},
				},
			}},

			// Ollama backend manager (opt-in)
			"ollama_manager": {Type: "ollama_manager", Enabled: true, Params: map[string]any{
				"enabled":            true,
				"auto_start":         true,  // try to start ollama if offline
				"auto_pull":          false, // WARNING: downloads models; keep opt-in
				"start_retries":      20,
				"start_retry_ms":     250,
				"pull_timeout_sec":   1800, // safety timeout for pull command
				"max_models_to_pull": 3,    // avoid pulling too much automatically
			}},
			"proposal_drive": {Type: "proposal_drive", Enabled: true, Params: map[string]any{
				"enabled":             true,
				"boost_per_pending":   0.08,
				"max_boost":           0.35,
				"notify_interval_sec": 300,
			}},
			"cooldown": {
				Type:    "cooldown",
				Enabled: true,
				Params: map[string]any{
					"seconds": 120,
				},
			},
			"say_energy_cost": {
				Type:    "say_energy_cost",
				Enabled: true,
				Params: map[string]any{
					"cost": 1.0,
				},
			},
			// Intent router (epigenetic, editable)
			"intent_router": {Type: "intent_router", Enabled: true, Params: map[string]any{
				// rules are evaluated by priority (high first)
				"rules": []any{
					map[string]any{
						"name":     "meta_self",
						"intent":   "META_BUNNY",
						"priority": 100,
						"contains": []any{"wie geht", "energie", "cooldown", "status", "was denkst"},
						"regex":    []any{`\bfühl\w*\b.*\bdu\b`, `\bwie\b.*\bgeht\b.*\bdir\b`},
					},
					map[string]any{
						"name":     "explicit_research",
						"intent":   "RESEARCH_CMD",
						"priority": 95,
						"contains": []any{"recherch", "im internet", "nachsehen", "quelle", "quellen"},
					},
					map[string]any{
						"name":     "opinion_request",
						"intent":   "OPINION",
						"priority": 90,
						"contains": []any{"meinung", "haltung", "position", "gut oder schlecht", "findest du"},
					},
					map[string]any{
						"name":     "external_fact",
						"intent":   "EXTERNAL_FACT",
						"priority": 70,
						"contains": []any{"wetter", "vorhersage", "wo liegt", "wo ist", "wer ist", "aktuell", "kurs", "preis"},
					},
					map[string]any{
						"name":     "task_tech",
						"intent":   "TASK_TECH",
						"priority": 50,
						"contains": []any{"patch", "github", "go ", "sqlite", "ollama", "fski"},
					},
				},
			}},
		},
		AffectDefsMap: map[string]AffectDef{
			"pain":   {Baseline: 0.05, DecayPerSec: 0.20, EnergyCoupling: 0.15},
			"unwell": {Baseline: 0.05, DecayPerSec: 0.15, EnergyCoupling: 0.10},
		},
	}
	if err := eg.Save(path); err != nil {
		return nil, err
	}
	return eg, nil
}

func (eg *Epigenome) Save(path string) error {
	b, err := json.MarshalIndent(eg, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (eg *Epigenome) ensureDefaults() (changed bool) {
	add := func(name string, spec *ModuleSpec) {
		if eg.Modules[name] == nil {
			eg.Modules[name] = spec
			changed = true
		}
	}
	add("auto_speak", &ModuleSpec{Type: "auto_speak", Enabled: true, Params: map[string]any{"cooldown_seconds": 18}})
	add("autonomy", &ModuleSpec{Type: "autonomy", Enabled: true, Params: map[string]any{
		"idle_seconds":     45,
		"min_talk_drive":   0.55,
		"cooldown_seconds": 60,
		"topic_k":          5,
	}})
	add("ollama_manager", &ModuleSpec{Type: "ollama_manager", Enabled: true, Params: map[string]any{
		"enabled":            true,
		"auto_start":         true,
		"auto_pull":          false,
		"start_retries":      20,
		"start_retry_ms":     250,
		"pull_timeout_sec":   1800,
		"max_models_to_pull": 3,
	}})
	add("semantic_memory", &ModuleSpec{Type: "semantic_memory", Enabled: true, Params: map[string]any{
		"enabled":             true,
		"max_writes_per_turn": 3,
		"max_reads_per_turn":  2,
		"write_rules":         []any{},
		"read_rules":          []any{},
	}})
	add("proposal_drive", &ModuleSpec{Type: "proposal_drive", Enabled: true, Params: map[string]any{
		"enabled":             true,
		"boost_per_pending":   0.08,
		"max_boost":           0.35,
		"notify_interval_sec": 300,
	}})

	def := func(k string, base, decay, coupling float64) {
		if _, ok := eg.AffectDefsMap[k]; !ok {
			eg.AffectDefsMap[k] = AffectDef{Baseline: base, DecayPerSec: decay, EnergyCoupling: coupling}
			changed = true
		}
	}
	def("pain", 0.05, 0.02, 0.0)
	def("sorrow", 0.02, 0.01, 0.0)
	def("unwell", 0.05, 0.02, 0.0)
	def("shame", 0.00, 0.03, 0.0)
	def("fear", 0.00, 0.03, 0.0)
	return changed
}

func (eg *Epigenome) ModuleEnabled(name string) bool {
	m := eg.Modules[name]
	return m != nil && m.Enabled
}

func (eg *Epigenome) ModuleParams(name string) map[string]any {
	m := eg.Modules[name]
	if m == nil || m.Params == nil {
		return map[string]any{}
	}
	return m.Params
}

func (eg *Epigenome) EnabledModuleNames() []string {
	var names []string
	for k, m := range eg.Modules {
		if m != nil && m.Enabled {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	return names
}

func (eg *Epigenome) Enable(name string, on bool) {
	m := eg.Modules[name]
	if m == nil {
		m = &ModuleSpec{Type: "unknown", Enabled: on, Params: map[string]any{}}
		eg.Modules[name] = m
		return
	}
	m.Enabled = on
}

func (eg *Epigenome) AddModule(name, typ string) error {
	if eg.Modules == nil {
		eg.Modules = map[string]*ModuleSpec{}
	}
	if _, exists := eg.Modules[name]; exists {
		return fmt.Errorf("module already exists: %s", name)
	}
	eg.Modules[name] = &ModuleSpec{
		Type:    typ,
		Enabled: true,
		Params:  map[string]any{},
	}
	switch typ {
	case "cooldown":
		eg.Modules[name].Params["seconds"] = 120
	case "say_energy_cost":
		eg.Modules[name].Params["cost"] = 1.0
	}
	return nil
}

func (eg *Epigenome) SetParam(name, key string, val any) error {
	m := eg.Modules[name]
	if m == nil {
		return fmt.Errorf("unknown module: %s", name)
	}
	if m.Params == nil {
		m.Params = map[string]any{}
	}
	m.Params[key] = val
	return nil
}

func (eg *Epigenome) AffectDefs() map[string]AffectDef {
	if eg.AffectDefsMap == nil {
		eg.AffectDefsMap = map[string]AffectDef{}
	}
	return eg.AffectDefsMap
}

func (eg *Epigenome) HeartbeatInterval() time.Duration {
	m := eg.Modules["heartbeat"]
	if m == nil || !m.Enabled {
		return 500 * time.Millisecond
	}
	ms := asFloat(m.Params["ms"], 500)
	if ms < 50 {
		ms = 50
	}
	return time.Duration(ms * float64(time.Millisecond))
}

func (eg *Epigenome) CooldownDuration() time.Duration {
	m := eg.Modules["cooldown"]
	if m == nil || !m.Enabled {
		return 2 * time.Minute
	}
	sec := asFloat(m.Params["seconds"], 120)
	if sec < 0 {
		sec = 0
	}
	return time.Duration(sec * float64(time.Second))
}

func (eg *Epigenome) AutoSpeakCooldownDuration() time.Duration {
	m := eg.Modules["auto_speak"]
	if m == nil || !m.Enabled {
		return 18 * time.Second
	}
	sec := asFloat(m.Params["cooldown_seconds"], 18)
	if sec < 5 {
		sec = 5
	}
	return time.Duration(sec * float64(time.Second))
}

func (eg *Epigenome) MemoryParams() (consolidateEvery int, contextTurns int, detailItems int, detailHalfLifeDays float64, episodeHalfLifeDays float64, latencyPainMs int, autoTune bool) {
	m := eg.Modules["memory"]
	if m == nil || !m.Enabled {
		return 16, 10, 6, 14.0, 120.0, 2500, true
	}
	consolidateEvery = int(asFloat(m.Params["consolidate_every_events"], 16))
	contextTurns = int(asFloat(m.Params["context_turns"], 10))
	detailItems = int(asFloat(m.Params["detail_items"], 6))
	detailHalfLifeDays = asFloat(m.Params["detail_half_life_days"], 14.0)
	episodeHalfLifeDays = asFloat(m.Params["episode_half_life_days"], 120.0)
	latencyPainMs = int(asFloat(m.Params["latency_pain_ms"], 2500))
	autoTune, _ = m.Params["auto_tune"].(bool)
	if consolidateEvery < 6 {
		consolidateEvery = 6
	}
	if consolidateEvery > 60 {
		consolidateEvery = 60
	}
	if contextTurns < 4 {
		contextTurns = 4
	}
	if contextTurns > 30 {
		contextTurns = 30
	}
	if detailItems < 0 {
		detailItems = 0
	}
	if detailItems > 20 {
		detailItems = 20
	}
	if detailHalfLifeDays < 1 {
		detailHalfLifeDays = 1
	}
	if detailHalfLifeDays > 365 {
		detailHalfLifeDays = 365
	}
	if episodeHalfLifeDays < 7 {
		episodeHalfLifeDays = 7
	}
	if latencyPainMs < 300 {
		latencyPainMs = 300
	}
	return
}

func (eg *Epigenome) SayEnergyCost() float64 {
	m := eg.Modules["say_energy_cost"]
	if m == nil || !m.Enabled {
		return 1.0
	}
	return asFloat(m.Params["cost"], 1.0)
}

func (eg *Epigenome) OllamaManagerParams() (enabled, autoStart, autoPull bool, startRetries int, startRetryMs int, pullTimeoutSec int, maxPull int) {
	m := eg.Modules["ollama_manager"]
	if m == nil || !m.Enabled {
		return false, false, false, 0, 0, 0, 0
	}
	if v, ok := m.Params["enabled"].(bool); ok {
		enabled = v
	} else {
		enabled = true
	}
	if v, ok := m.Params["auto_start"].(bool); ok {
		autoStart = v
	} else {
		autoStart = true
	}
	if v, ok := m.Params["auto_pull"].(bool); ok {
		autoPull = v
	} else {
		autoPull = false
	}
	startRetries = int(asFloat(m.Params["start_retries"], 20))
	startRetryMs = int(asFloat(m.Params["start_retry_ms"], 250))
	pullTimeoutSec = int(asFloat(m.Params["pull_timeout_sec"], 1800))
	maxPull = int(asFloat(m.Params["max_models_to_pull"], 3))

	if startRetries < 0 {
		startRetries = 0
	}
	if startRetries > 200 {
		startRetries = 200
	}
	if startRetryMs < 50 {
		startRetryMs = 50
	}
	if startRetryMs > 5000 {
		startRetryMs = 5000
	}
	if pullTimeoutSec < 60 {
		pullTimeoutSec = 60
	}
	if pullTimeoutSec > 7200 {
		pullTimeoutSec = 7200
	}
	if maxPull < 0 {
		maxPull = 0
	}
	if maxPull > 20 {
		maxPull = 20
	}
	return
}

func (eg *Epigenome) ProposalDriveParams() (enabled bool, boostPer float64, maxBoost float64, notifyIntervalSec float64) {
	m := eg.Modules["proposal_drive"]
	if m == nil || !m.Enabled {
		return false, 0, 0, 0
	}
	if v, ok := m.Params["enabled"].(bool); ok {
		enabled = v
	} else {
		enabled = true
	}
	boostPer = asFloat(m.Params["boost_per_pending"], 0.08)
	maxBoost = asFloat(m.Params["max_boost"], 0.35)
	notifyIntervalSec = asFloat(m.Params["notify_interval_sec"], 300)
	if boostPer < 0 {
		boostPer = 0
	}
	if maxBoost < 0 {
		maxBoost = 0
	}
	if maxBoost > 0.9 {
		maxBoost = 0.9
	}
	if notifyIntervalSec < 30 {
		notifyIntervalSec = 30
	}
	if notifyIntervalSec > 3600 {
		notifyIntervalSec = 3600
	}
	return
}

type IntentRule struct {
	Name     string
	Intent   string
	Priority int
	Contains []string
	Regex    []string
}

func (eg *Epigenome) IntentRules() []IntentRule {
	m := eg.Modules["intent_router"]
	if m == nil || !m.Enabled {
		return nil
	}
	raw, ok := m.Params["rules"]
	if !ok || raw == nil {
		return nil
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]IntentRule, 0, len(arr))
	for _, it := range arr {
		mm, ok := it.(map[string]any)
		if !ok {
			continue
		}
		r := IntentRule{}
		if v, ok := mm["name"].(string); ok {
			r.Name = v
		}
		if v, ok := mm["intent"].(string); ok {
			r.Intent = v
		}
		r.Priority = int(asFloat(mm["priority"], 0))
		if c, ok := mm["contains"].([]any); ok {
			for _, x := range c {
				if s, ok := x.(string); ok && s != "" {
					r.Contains = append(r.Contains, s)
				}
			}
		}
		if c, ok := mm["regex"].([]any); ok {
			for _, x := range c {
				if s, ok := x.(string); ok && s != "" {
					r.Regex = append(r.Regex, s)
				}
			}
		}
		if r.Intent != "" {
			out = append(out, r)
		}
	}
	for i := 0; i < len(out); i++ {
		for j := i + 1; j < len(out); j++ {
			if out[j].Priority > out[i].Priority {
				out[i], out[j] = out[j], out[i]
			}
		}
	}
	return out
}

func (eg *Epigenome) Values() map[string]float64 {
	m := eg.Modules["values"]
	out := map[string]float64{}
	if m == nil || !m.Enabled {
		return out
	}
	for k, v := range m.Params {
		out[k] = asFloat(v, 0)
	}
	return out
}

func (eg *Epigenome) StanceParams() (halfLifeDays float64, minConfidence float64, autoUpdate bool) {
	m := eg.Modules["stance"]
	if m == nil || !m.Enabled {
		return 60.0, 0.35, true
	}
	halfLifeDays = asFloat(m.Params["half_life_days"], 60.0)
	minConfidence = asFloat(m.Params["min_confidence"], 0.35)
	autoUpdate, _ = m.Params["auto_update"].(bool)
	if halfLifeDays < 7 {
		halfLifeDays = 7
	}
	if minConfidence < 0 {
		minConfidence = 0
	}
	if minConfidence > 1 {
		minConfidence = 1
	}
	return
}

func (eg *Epigenome) ScoutParams() (intervalSec int, minCuriosity float64, maxPerHour int, enabled bool) {
	m := eg.Modules["scout"]
	if m == nil || !m.Enabled {
		return 45, 0.55, 24, false
	}
	enabled = true
	intervalSec = int(asFloat(m.Params["interval_seconds"], 45))
	minCuriosity = asFloat(m.Params["min_curiosity"], 0.55)
	maxPerHour = int(asFloat(m.Params["max_per_hour"], 24))
	if intervalSec < 10 {
		intervalSec = 10
	}
	if intervalSec > 600 {
		intervalSec = 600
	}
	if minCuriosity < 0 {
		minCuriosity = 0
	}
	if minCuriosity > 1 {
		minCuriosity = 1
	}
	if maxPerHour < 1 {
		maxPerHour = 1
	}
	if maxPerHour > 240 {
		maxPerHour = 240
	}
	return
}

func (eg *Epigenome) DaydreamParams() (intervalSec int, minCuriosity float64, minEnergy float64, visualWeight float64, enabled bool) {
	m := eg.Modules["daydream"]
	if m == nil || !m.Enabled {
		return 20, 0.45, 8, 0.55, false
	}
	enabled = true
	intervalSec = int(asFloat(m.Params["interval_seconds"], 20))
	minCuriosity = asFloat(m.Params["min_curiosity"], 0.45)
	minEnergy = asFloat(m.Params["min_energy"], 8)
	visualWeight = asFloat(m.Params["visual_weight"], 0.55)
	if intervalSec < 5 {
		intervalSec = 5
	}
	if intervalSec > 600 {
		intervalSec = 600
	}
	if minCuriosity < 0 {
		minCuriosity = 0
	}
	if minCuriosity > 1 {
		minCuriosity = 1
	}
	if minEnergy < 0 {
		minEnergy = 0
	}
	if minEnergy > 100 {
		minEnergy = 100
	}
	if visualWeight < 0 {
		visualWeight = 0
	}
	if visualWeight > 1 {
		visualWeight = 1
	}
	return
}

func (eg *Epigenome) CriticEnabled() bool {
	m := eg.Modules["critic"]
	if m == nil || !m.Enabled {
		return false
	}
	enabled, ok := m.Params["enabled"].(bool)
	if ok {
		return enabled
	}
	return true
}

func (eg *Epigenome) Lang() string {
	m := eg.Modules["locale"]
	if m == nil || !m.Enabled {
		return "de"
	}
	if s, ok := m.Params["lang"].(string); ok && s != "" {
		return s
	}
	return "de"
}

func (eg *Epigenome) EnergyMax() float64 {
	m := eg.Modules["energy"]
	if m == nil || !m.Enabled {
		return 100
	}
	return asFloat(m.Params["max"], 100)
}

func (eg *Epigenome) UtteranceBannedPhrases() []string {
	m := eg.Modules["utterance_filter"]
	if m == nil || !m.Enabled {
		return nil
	}
	raw, ok := m.Params["banned_phrases"]
	if !ok || raw == nil {
		return nil
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, v := range arr {
		if s, ok := v.(string); ok && s != "" {
			out = append(out, s)
		}
	}
	return out
}

type DrivesV1Params struct {
	DiskPath                      string
	DiskTargetBytes               float64
	RamTargetBytes                float64
	LatencyTargetMs               float64
	Wdisk, Wram, Wcpu, Wlat, Werr float64
	Kdisk, Kram, Kcpu             float64
	TauSocialSec                  float64
	EmaUser                       float64
	EmaCaught                     float64
	HelpMinIntervalSec            float64
	Enabled                       bool
}

type SemWriteRule struct {
	Name         string
	Regex        string
	Subject      string
	Predicate    string
	Object       string
	Confidence   float64
	Salience     float64
	HalfLifeDays float64
	Source       string
	Ack          string
}

type SemReadRule struct {
	Name          string
	Regex         string
	Subject       string
	Predicate     string
	AnswerFound   string
	AnswerMissing string
}

func (eg *Epigenome) SemanticMemoryRules() (enabled bool, maxWrites int, maxReads int, wr []SemWriteRule, rr []SemReadRule) {
	m := eg.Modules["semantic_memory"]
	if m == nil || !m.Enabled {
		return false, 0, 0, nil, nil
	}
	if v, ok := m.Params["enabled"].(bool); ok {
		enabled = v
	} else {
		enabled = true
	}
	maxWrites = int(asFloat(m.Params["max_writes_per_turn"], 3))
	maxReads = int(asFloat(m.Params["max_reads_per_turn"], 2))
	if maxWrites < 0 {
		maxWrites = 0
	}
	if maxWrites > 20 {
		maxWrites = 20
	}
	if maxReads < 0 {
		maxReads = 0
	}
	if maxReads > 20 {
		maxReads = 20
	}

	if arr, ok := m.Params["write_rules"].([]any); ok {
		for _, it := range arr {
			mm, _ := it.(map[string]any)
			if mm == nil {
				continue
			}
			r := SemWriteRule{
				Name:         asString(mm["name"], ""),
				Regex:        asString(mm["regex"], ""),
				Subject:      asString(mm["subject"], ""),
				Predicate:    asString(mm["predicate"], ""),
				Object:       asString(mm["object"], ""),
				Confidence:   asFloat(mm["confidence"], 0.7),
				Salience:     asFloat(mm["salience"], 0.5),
				HalfLifeDays: asFloat(mm["half_life_days"], 365),
				Source:       asString(mm["source"], "user"),
				Ack:          asString(mm["ack"], ""),
			}
			if r.Regex != "" && r.Subject != "" && r.Predicate != "" && r.Object != "" {
				wr = append(wr, r)
			}
		}
	}

	if arr, ok := m.Params["read_rules"].([]any); ok {
		for _, it := range arr {
			mm, _ := it.(map[string]any)
			if mm == nil {
				continue
			}
			r := SemReadRule{
				Name:          asString(mm["name"], ""),
				Regex:         asString(mm["regex"], ""),
				Subject:       asString(mm["subject"], ""),
				Predicate:     asString(mm["predicate"], ""),
				AnswerFound:   asString(mm["answer_found"], ""),
				AnswerMissing: asString(mm["answer_missing"], ""),
			}
			if r.Regex != "" && r.Subject != "" && r.Predicate != "" {
				rr = append(rr, r)
			}
		}
	}
	return
}

func (eg *Epigenome) DrivesV1() DrivesV1Params {
	m := eg.Modules["drives_v1"]
	if m == nil || !m.Enabled {
		return DrivesV1Params{Enabled: false}
	}
	p := DrivesV1Params{Enabled: true}
	p.DiskPath, _ = m.Params["disk_path"].(string)
	if p.DiskPath == "" {
		p.DiskPath = "C:\\"
	}
	p.DiskTargetBytes = asFloat(m.Params["disk_target_bytes"], 1.0e10)
	p.RamTargetBytes = asFloat(m.Params["ram_target_bytes"], 3.0e9)
	p.LatencyTargetMs = asFloat(m.Params["latency_target_ms"], 2500)
	p.Wdisk = asFloat(m.Params["w_disk"], 0.30)
	p.Wram = asFloat(m.Params["w_ram"], 0.30)
	p.Wcpu = asFloat(m.Params["w_cpu"], 0.20)
	p.Wlat = asFloat(m.Params["w_lat"], 0.15)
	p.Werr = asFloat(m.Params["w_err"], 0.05)
	p.Kdisk = asFloat(m.Params["k_disk"], 4.0)
	p.Kram = asFloat(m.Params["k_ram"], 5.0)
	p.Kcpu = asFloat(m.Params["k_cpu"], 3.0)
	p.TauSocialSec = asFloat(m.Params["tau_social_seconds"], 1200)
	p.EmaUser = asFloat(m.Params["ema_user_reward"], 0.12)
	p.EmaCaught = asFloat(m.Params["ema_caught"], 0.20)
	p.HelpMinIntervalSec = asFloat(m.Params["help_min_interval_seconds"], 180)
	return p
}

func asString(v any, d string) string {
	if s, ok := v.(string); ok {
		return s
	}
	return d
}

func asFloat(v any, def float64) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case json.Number:
		f, _ := x.Float64()
		if f == 0 {
			return def
		}
		return f
	default:
		return def
	}
}

func (eg *Epigenome) IntentNBParams() (enabled bool, minTokens int, threshold float64, alpha float64) {
	m := eg.Modules["intent_nb"]
	if m == nil || !m.Enabled {
		return false, 2, 0.72, 1.0
	}
	if v, ok := m.Params["enabled"].(bool); ok {
		enabled = v
	} else {
		enabled = true
	}
	minTokens = int(asFloat(m.Params["min_tokens"], 2))
	threshold = asFloat(m.Params["threshold"], 0.72)
	alpha = asFloat(m.Params["alpha"], 1.0)
	if minTokens < 1 {
		minTokens = 1
	}
	if minTokens > 10 {
		minTokens = 10
	}
	if threshold < 0 {
		threshold = 0
	}
	if threshold > 0.99 {
		threshold = 0.99
	}
	if alpha <= 0 {
		alpha = 1.0
	}
	return
}

func (eg *Epigenome) InfoGateParams() (enabled bool, minInfo float64, idfThreshold float64, idf2charThreshold float64, stopwordRatio float64, minTokens int, warmupMinDocs int, stopwordMinDf int) {
	m := eg.Modules["info_gate"]
	if m == nil || !m.Enabled {
		return false, 0.33, 1.0, 3.0, 0.35, 1, 30, 8
	}
	if v, ok := m.Params["enabled"].(bool); ok {
		enabled = v
	} else {
		enabled = true
	}
	minInfo = asFloat(m.Params["min_info"], 0.33)
	idfThreshold = asFloat(m.Params["idf_threshold"], 1.0)
	idf2charThreshold = asFloat(m.Params["idf_2char_threshold"], 3.0)
	stopwordRatio = asFloat(m.Params["stopword_ratio"], 0.35)
	minTokens = int(asFloat(m.Params["min_tokens"], 1))
	warmupMinDocs = int(asFloat(m.Params["warmup_min_docs"], 30))
	stopwordMinDf = int(asFloat(m.Params["stopword_min_df"], 8))
	if minInfo < 0 {
		minInfo = 0
	}
	if minInfo > 0.95 {
		minInfo = 0.95
	}
	if idfThreshold < 0.1 {
		idfThreshold = 0.1
	}
	if idf2charThreshold < idfThreshold {
		idf2charThreshold = idfThreshold
	}
	if stopwordRatio < 0.05 {
		stopwordRatio = 0.05
	}
	if stopwordRatio > 0.90 {
		stopwordRatio = 0.90
	}
	if minTokens < 1 {
		minTokens = 1
	}
	if minTokens > 10 {
		minTokens = 10
	}
	if warmupMinDocs < 0 {
		warmupMinDocs = 0
	}
	if warmupMinDocs > 500 {
		warmupMinDocs = 500
	}
	if stopwordMinDf < 1 {
		stopwordMinDf = 1
	}
	if stopwordMinDf > 1000 {
		stopwordMinDf = 1000
	}
	return
}

func (eg *Epigenome) ModelFor(area string, fallback string) string {
	m := eg.Modules["models"]
	if m == nil || !m.Enabled {
		if fallback != "" {
			return fallback
		}
		return "llama3.1:8b"
	}
	area = strings.ToLower(strings.TrimSpace(area))
	if v, ok := m.Params[area].(string); ok && v != "" {
		return v
	}
	if v, ok := m.Params["default"].(string); ok && v != "" {
		return v
	}
	if fallback != "" {
		return fallback
	}
	return "llama3.1:8b"
}
