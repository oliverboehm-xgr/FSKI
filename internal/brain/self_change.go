package brain

import (
	"crypto/sha1"
	"database/sql"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
)

// CommitSelfChange is the ONLY entry point that is allowed to persist autonomous changes.
//
// Patch #1 scope:
//   - evaluate axioms (lexicographic)
//   - charge metabolic cost (energy) + throttle counter
//   - write an immutable log record (self_changes)
//
// Wiring the actual mutation targets (epigenome apply, concepts, LoRA jobs, code patches)
// is done in later patches.
func CommitSelfChange(db *sql.DB, eg *epi.Epigenome, body any, ws *Workspace, ch SelfChange) (AxiomDecision, float64) {
	dec := EvaluateAxioms(ch)
	if db == nil {
		return dec, 0
	}

	base, cooldownSec := selfChangeBaseCost(eg, ch.Kind)
	count := bumpSelfChangeCounter(db)
	k := selfChangeProgressiveK(eg)
	mult := 1.0 + (float64(count-1)*float64(count-1))*k
	cost := base * mult

	if !dec.Allowed {
		// Blocked attempts still cost a bit (prevents thrash).
		cost = clamp(cost*0.25, 0.1, base)
		cooldownSec = int(float64(cooldownSec) * 0.3)
	}

	chargeSelfChange(body, ws, cost)
	rollbackKey := makeRollbackKey(ch)
	insertSelfChangeLog(db, ch, dec, cost, rollbackKey)
	_ = cooldownSec // cooldown wiring is done in patch #3 (BodyState is in cmd package).

	return dec, cost
}

func selfChangeProgressiveK(eg *epi.Epigenome) float64 {
	k := 0.08
	if eg == nil {
		return k
	}
	m := eg.Modules["self_change_cost"]
	if m == nil || !m.Enabled || m.Params == nil {
		return k
	}
	return asFloatAny(m.Params["progressive_k"], k)
}

func selfChangeBaseCost(eg *epi.Epigenome, kind string) (base float64, cooldownSec int) {
	base = 1.0
	cooldownSec = 20

	k := strings.ToLower(strings.TrimSpace(kind))
	switch k {
	case "concept", "axiom":
		base = 0.6
		cooldownSec = 10
	case "policy", "epigenome":
		base = 2.0
		cooldownSec = 40
	case "lora":
		base = 4.0
		cooldownSec = 90
	case "code":
		base = 6.0
		cooldownSec = 120
	}
	if eg == nil {
		return base, cooldownSec
	}
	return eg.SelfChangeCostParams(k, base, cooldownSec)
}

func chargeSelfChange(body any, ws *Workspace, energyCost float64) {
	if energyCost <= 0 {
		return
	}
	cur := epi.ExtractEnergy(body)
	cur -= energyCost
	if cur < 0 {
		cur = 0
	}
	epi.InjectEnergy(body, cur)
	if ws != nil {
		ws.EnergyHint = cur
	}
}

func makeRollbackKey(ch SelfChange) string {
	h := sha1.Sum([]byte(ch.Kind + "|" + ch.Target + "|" + ch.DeltaJSON))
	return hex.EncodeToString(h[:])
}

func insertSelfChangeLog(db *sql.DB, ch SelfChange, dec AxiomDecision, energyCost float64, rollbackKey string) {
	now := time.Now().Format(time.RFC3339)
	allowed := 0
	if dec.Allowed {
		allowed = 1
	}
	block := dec.BlockAxiom
	if block < 0 {
		block = 0
	}
	_, _ = db.Exec(`INSERT INTO self_changes(created_at,kind,target,delta_json,axiom_goal,allowed,axiom_block,risk,energy_cost,note,rollback_key)
		VALUES(?,?,?,?,?,?,?,?,?,?,?)`,
		now,
		strings.TrimSpace(ch.Kind),
		strings.TrimSpace(ch.Target),
		strings.TrimSpace(ch.DeltaJSON),
		ch.AxiomGoal,
		allowed,
		block,
		string(dec.Risk),
		energyCost,
		strings.TrimSpace(ch.Note),
		rollbackKey,
	)
}

func bumpSelfChangeCounter(db *sql.DB) int {
	if db == nil {
		return 1
	}
	key := "self_change:count_24h"
	keyTs := "self_change:count_24h_ts"
	now := time.Now()

	var rawTs string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, keyTs).Scan(&rawTs)
	if t, err := time.Parse(time.RFC3339, strings.TrimSpace(rawTs)); err == nil {
		if now.Sub(t) > 24*time.Hour {
			// reset
			_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at`, key, "0", now.Format(time.RFC3339))
			_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at`, keyTs, now.Format(time.RFC3339), now.Format(time.RFC3339))
		}
	} else {
		// init
		_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at`, key, "0", now.Format(time.RFC3339))
		_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at`, keyTs, now.Format(time.RFC3339), now.Format(time.RFC3339))
	}

	var raw string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, key).Scan(&raw)
	n, _ := strconv.Atoi(strings.TrimSpace(raw))
	n++
	_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at`, key, fmt.Sprintf("%d", n), now.Format(time.RFC3339))
	return n
}

func asFloatAny(v any, fallback float64) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case string:
		f, err := strconv.ParseFloat(strings.TrimSpace(t), 64)
		if err == nil {
			return f
		}
	}
	return fallback
}

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
