package brain

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
)

type EvolutionParams struct {
	Enabled        bool
	IntervalHours  int
	WindowHours    int
	ForkCount      int
	BudgetSeconds  int
	Alpha          float64
	Beta           float64
	Gamma          float64
	Delta          float64
	Epsilon        float64
	ProposalPrefix string
}

type evolutionMetrics struct {
	UserReward float64
	Evidence   float64
	Cost       float64
	Spam       float64
	Coherence  float64
}

type evolutionCandidate struct {
	Index   int
	Title   string
	Patch   map[string]any
	Fitness float64
	evolutionMetrics
}

func TickEvolutionTournament(db *sql.DB, eg *epi.Epigenome, now time.Time) (bool, string) {
	if db == nil || eg == nil {
		return false, ""
	}
	p := LoadEvolutionTournamentParams(eg)
	if !p.Enabled {
		return false, ""
	}
	if p.IntervalHours <= 0 {
		p.IntervalHours = 24
	}
	if ts, ok := kvTime(db, "evolution:last_run_at"); ok {
		if now.Sub(ts) < time.Duration(p.IntervalHours)*time.Hour {
			return false, ""
		}
	}

	windowStart := now.Add(-time.Duration(p.WindowHours) * time.Hour)
	base := loadEvolutionMetrics(db, windowStart, now)
	cands := buildEvolutionCandidates(eg, p, base)
	if len(cands) == 0 {
		return false, ""
	}

	winner := cands[0]
	for i := 1; i < len(cands); i++ {
		if cands[i].Fitness > winner.Fitness {
			winner = cands[i]
		}
	}

	notes := fmt.Sprintf("window=%s..%s base={reward=%.3f evidence=%.3f cost=%.3f spam=%.3f coherence=%.3f}",
		windowStart.Format(time.RFC3339), now.Format(time.RFC3339), base.UserReward, base.Evidence, base.Cost, base.Spam, base.Coherence)
	runID := insertEvolutionRun(db, now, windowStart, p, winner, notes)
	for _, c := range cands {
		insertEvolutionCandidate(db, runID, c, now)
	}

	patchBytes, _ := json.Marshal(winner.Patch)
	title := strings.TrimSpace(p.ProposalPrefix) + ".winner.r" + fmt.Sprintf("%d", runID)
	if strings.TrimSpace(p.ProposalPrefix) == "" {
		title = "evolution_tournament.winner.r" + fmt.Sprintf("%d", runID)
	}
	_, _ = InsertEpigenomeProposal(db, title, string(patchBytes), fmt.Sprintf("auto tournament winner idx=%d score=%.3f", winner.Index, winner.Fitness))
	setKV(db, "evolution:last_run_at", now.Format(time.RFC3339))

	msg := fmt.Sprintf("Evolution-Tournament: %d Kandidaten evaluiert. Sieger #%d (Fitness %.3f). Vorschlag als /epi proposal angelegt.", len(cands), winner.Index, winner.Fitness)
	return true, msg
}

func LoadEvolutionTournamentParams(eg *epi.Epigenome) EvolutionParams {
	enabled, interval, window, forks, budget, a, b, g, d, e, prefix := eg.EvolutionTournamentParams()
	if interval < 1 {
		interval = 24
	}
	if window < 1 {
		window = 24
	}
	if forks < 2 {
		forks = 2
	}
	if forks > 16 {
		forks = 16
	}
	if budget < 30 {
		budget = 30
	}
	return EvolutionParams{Enabled: enabled, IntervalHours: interval, WindowHours: window, ForkCount: forks, BudgetSeconds: budget, Alpha: a, Beta: b, Gamma: g, Delta: d, Epsilon: e, ProposalPrefix: prefix}
}

func loadEvolutionMetrics(db *sql.DB, from, to time.Time) evolutionMetrics {
	var m evolutionMetrics
	_ = db.QueryRow(`SELECT COALESCE(AVG(value),0) FROM ratings WHERE created_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&m.UserReward)
	var sourceN, factN int
	_ = db.QueryRow(`SELECT COUNT(*) FROM sources WHERE fetched_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&sourceN)
	_ = db.QueryRow(`SELECT COUNT(*) FROM facts WHERE updated_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&factN)
	m.Evidence = evClamp01(float64(sourceN)/40.0 + float64(factN)/30.0)
	var webN, msgN int
	_ = db.QueryRow(`SELECT COUNT(*) FROM events WHERE channel='web' AND created_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&webN)
	_ = db.QueryRow(`SELECT COUNT(*) FROM messages WHERE created_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&msgN)
	m.Cost = evClamp01(float64(webN)/45.0 + float64(msgN)/350.0)
	var caught int
	_ = db.QueryRow(`SELECT COUNT(*) FROM caught_events WHERE created_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&caught)
	m.Spam = evClamp01(float64(caught) / 12.0)
	var autoDown int
	_ = db.QueryRow(`SELECT COUNT(*)
		FROM ratings r JOIN message_meta mm ON mm.message_id=r.message_id
		WHERE mm.kind='auto' AND r.value<0 AND r.created_at BETWEEN ? AND ?`, from.Format(time.RFC3339), to.Format(time.RFC3339)).Scan(&autoDown)
	m.Coherence = evClamp01((1.0-m.Spam)*0.7 + evClamp01(1.0-float64(autoDown)/8.0)*0.3)
	return m
}

func buildEvolutionCandidates(eg *epi.Epigenome, p EvolutionParams, base evolutionMetrics) []evolutionCandidate {
	out := make([]evolutionCandidate, 0, p.ForkCount)
	for i := 0; i < p.ForkCount; i++ {
		drift := (float64(i) - float64(p.ForkCount-1)/2.0) / math.Max(1.0, float64(p.ForkCount-1))
		cand := evolutionCandidate{Index: i + 1}
		cand.Title = fmt.Sprintf("evolution.candidate.%02d", cand.Index)

		minTalk := evClamp01(egFloat(eg, "autonomy", "min_talk_drive", 0.55) + 0.08*drift)
		scoutMin := evClamp01(egFloat(eg, "scout", "min_curiosity", 0.55) + 0.10*drift)
		friction := evClamp01(egFloat(eg, "proposal_engine", "friction_threshold", 0.55) - 0.08*drift)
		daydreamSec := clampFloor(egFloat(eg, "daydream", "interval_seconds", 20)+6.0*drift, 8)

		cand.Patch = map[string]any{"modules": map[string]any{
			"autonomy":        map[string]any{"params": map[string]any{"min_talk_drive": round3(minTalk)}},
			"scout":           map[string]any{"params": map[string]any{"min_curiosity": round3(scoutMin)}},
			"proposal_engine": map[string]any{"params": map[string]any{"friction_threshold": round3(friction)}},
			"daydream":        map[string]any{"params": map[string]any{"interval_seconds": int(daydreamSec)}},
		}}

		cand.UserReward = evClamp01((base.UserReward+1.0)/2.0 + 0.10*(0.5-math.Abs(drift)))
		cand.Evidence = evClamp01(base.Evidence + 0.12*max0(drift))
		cand.Cost = evClamp01(base.Cost + 0.18*max0(-drift))
		cand.Spam = evClamp01(base.Spam + 0.15*max0(-drift))
		cand.Coherence = evClamp01(base.Coherence + 0.09*(0.5-math.Abs(drift)))
		cand.Fitness = p.Alpha*cand.UserReward + p.Beta*cand.Evidence - p.Gamma*cand.Cost - p.Delta*cand.Spam + p.Epsilon*cand.Coherence
		out = append(out, cand)
	}
	return out
}

func insertEvolutionRun(db *sql.DB, now, start time.Time, p EvolutionParams, winner evolutionCandidate, notes string) int64 {
	w := map[string]float64{"alpha": p.Alpha, "beta": p.Beta, "gamma": p.Gamma, "delta": p.Delta, "epsilon": p.Epsilon}
	b, _ := json.Marshal(w)
	res, err := db.Exec(`INSERT INTO evolution_runs(created_at,window_start,window_end,fork_count,budget_seconds,weights_json,winner_index,winner_score,notes) VALUES(?,?,?,?,?,?,?,?,?)`,
		now.Format(time.RFC3339), start.Format(time.RFC3339), now.Format(time.RFC3339), p.ForkCount, p.BudgetSeconds, string(b), winner.Index, winner.Fitness, notes)
	if err != nil {
		return 0
	}
	id, _ := res.LastInsertId()
	return id
}

func insertEvolutionCandidate(db *sql.DB, runID int64, c evolutionCandidate, now time.Time) {
	b, _ := json.Marshal(c.Patch)
	_, _ = db.Exec(`INSERT INTO evolution_candidates(run_id,candidate_index,title,patch_json,user_reward,evidence,cost,spam,coherence,fitness,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)`,
		runID, c.Index, c.Title, string(b), c.UserReward, c.Evidence, c.Cost, c.Spam, c.Coherence, c.Fitness, now.Format(time.RFC3339))
}

func kvTime(db *sql.DB, key string) (time.Time, bool) {
	var v string
	if err := db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, key).Scan(&v); err != nil {
		return time.Time{}, false
	}
	t, err := time.Parse(time.RFC3339, strings.TrimSpace(v))
	if err != nil {
		return time.Time{}, false
	}
	return t, true
}

func egFloat(eg *epi.Epigenome, module, key string, def float64) float64 {
	m := eg.Modules[module]
	if m == nil {
		return def
	}
	return floatFromAny(m.Params[key], def)
}

func evClamp01(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func clampFloor(v float64, min float64) float64 {
	if v < min {
		return min
	}
	return v
}

func round3(v float64) float64 { return math.Round(v*1000) / 1000 }

func max0(v float64) float64 {
	if v < 0 {
		return 0
	}
	return v
}

func floatFromAny(v any, def float64) float64 {
	switch t := v.(type) {
	case float64:
		return t
	case float32:
		return float64(t)
	case int:
		return float64(t)
	case int64:
		return float64(t)
	case int32:
		return float64(t)
	case uint:
		return float64(t)
	case uint64:
		return float64(t)
	case json.Number:
		f, err := t.Float64()
		if err == nil {
			return f
		}
	}
	return def
}
