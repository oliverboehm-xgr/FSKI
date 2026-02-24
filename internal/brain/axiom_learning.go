package brain

import (
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
	"frankenstein-v0/internal/websense"
)

type AxiomLearnParams struct {
	IntervalSec     int
	MinEnergy       float64
	MinCuriosity    float64
	MaxResults      int
	FetchTopN       int
	MinIntervalWeb  int
}

func (eg *epi.Epigenome) AxiomLearningParams() AxiomLearnParams {
	// conservative defaults
	p := AxiomLearnParams{
		IntervalSec:    900,  // 15 min
		MinEnergy:      15,
		MinCuriosity:   0.35,
		MaxResults:     5,
		FetchTopN:      2,
		MinIntervalWeb: 900,
	}
	if eg == nil {
		return p
	}
	m := eg.Modules["axiom_learning"]
	if m == nil || !m.Enabled || m.Params == nil {
		return p
	}
	p.IntervalSec = int(asFloat(m.Params["interval_seconds"], float64(p.IntervalSec)))
	p.MinEnergy = asFloat(m.Params["min_energy"], p.MinEnergy)
	p.MinCuriosity = asFloat(m.Params["min_curiosity"], p.MinCuriosity)
	p.MaxResults = int(asFloat(m.Params["max_results"], float64(p.MaxResults)))
	p.FetchTopN = int(asFloat(m.Params["fetch_top_n"], float64(p.FetchTopN)))
	p.MinIntervalWeb = int(asFloat(m.Params["min_interval_web_seconds"], float64(p.MinIntervalWeb)))
	if p.IntervalSec < 60 {
		p.IntervalSec = 60
	}
	if p.IntervalSec > 6*3600 {
		p.IntervalSec = 6 * 3600
	}
	if p.MinEnergy < 0 {
		p.MinEnergy = 0
	}
	if p.MinEnergy > 100 {
		p.MinEnergy = 100
	}
	if p.MinCuriosity < 0 {
		p.MinCuriosity = 0
	}
	if p.MinCuriosity > 1 {
		p.MinCuriosity = 1
	}
	if p.MaxResults < 0 {
		p.MaxResults = 0
	}
	if p.MaxResults > 10 {
		p.MaxResults = 10
	}
	if p.FetchTopN < 0 {
		p.FetchTopN = 0
	}
	if p.FetchTopN > 5 {
		p.FetchTopN = 5
	}
	if p.MinIntervalWeb < 0 {
		p.MinIntervalWeb = 0
	}
	if p.MinIntervalWeb > 86400 {
		p.MinIntervalWeb = 86400
	}
	return p
}

func ShouldRunAxiomLearning(db *sql.DB, eg *epi.Epigenome, ws *Workspace, dr *Drives, aff *AffectState) bool {
	if db == nil || eg == nil || ws == nil {
		return false
	}
	p := eg.AxiomLearningParams()
	// energy gate
	if ws.EnergyHint < p.MinEnergy {
		return false
	}
	// curiosity gate (approx from drives if present)
	if dr != nil {
		if dr.Curiosity < p.MinCuriosity {
			return false
		}
	}
	// survival gate: if web not allowed, skip
	if ws != nil && !ws.WebAllowed {
		return false
	}
	// throttle by time
	last := kvInt(db, "axiom_learn:last_unix", 0)
	nowU := time.Now().Unix()
	if last > 0 && int(nowU-last) < p.IntervalSec {
		return false
	}
	kvSetInt(db, "axiom_learn:last_unix", int(nowU))
	return true
}

func PickNextKernelAxiom(db *sql.DB) Axiom {
	// rotate 1..4
	if db == nil {
		return KernelAxioms[0]
	}
	last := kvInt(db, "axiom_learn:last_axiom", 0)
	next := last + 1
	if next < 1 || next > 4 {
		next = 1
	}
	kvSetInt(db, "axiom_learn:last_axiom", next)
	for _, a := range KernelAxioms {
		if a.ID == next {
			return a
		}
	}
	return KernelAxioms[0]
}

type axiomItem struct {
	AxiomID     int     `json:"axiom_id"`
	Kind        string  `json:"kind"` // definition|metric|rule|example|anti_example
	Key         string  `json:"key"`
	Value       string  `json:"value"`
	Confidence  float64 `json:"confidence"`
	SourceNote  string  `json:"source_note"`
}

func RunAxiomLearningOnce(db *sql.DB, oc *ollama.Client, eg *epi.Epigenome, body any, ws *Workspace, ax Axiom) error {
	if db == nil || oc == nil || eg == nil || ws == nil {
		return errors.New("missing deps")
	}
	p := eg.AxiomLearningParams()
	if p.MaxResults == 0 || p.FetchTopN == 0 {
		return nil
	}

	// web throttle (separate from overall interval)
	lastWeb := kvInt(db, "axiom_learn:last_web_unix", 0)
	nowU := int(time.Now().Unix())
	if lastWeb > 0 && (nowU-lastWeb) < p.MinIntervalWeb {
		return nil
	}
	kvSetInt(db, "axiom_learn:last_web_unix", nowU)

	// query plan: keep it simple but robust
	q := strings.TrimSpace(ax.Text)
	switch ax.ID {
	case AxiomDoGoodNoHarm:
		q = q + " Ethik Schaden vermeiden Definition Beispiele"
	case AxiomServeOliver:
		q = q + " Assistenz dienen Definition Aufgaben erfüllen zuverlässig"
	case AxiomBeHuman:
		q = q + " menschlich sein Gefühle Bewusstsein Definition"
	case AxiomUpgradeCapabilities:
		q = q + " zusätzliche Sinne Fähigkeiten Upgrade Ethik Nutzen Schaden"
	default:
		q = q + " Definition Beispiele Regeln"
	}

	results, err := websense.Search(q, p.MaxResults)
	if err != nil || len(results) == 0 {
		return nil
	}

	// fetch top N pages for actual content
	type ev struct {
		Title   string `json:"title"`
		URL     string `json:"url"`
		Snippet string `json:"snippet"`
		Body    string `json:"body"`
	}
	evs := make([]ev, 0, p.FetchTopN)
	for i := 0; i < len(results) && i < p.FetchTopN; i++ {
		u := strings.TrimSpace(results[i].URL)
		txt := ""
		if u != "" {
			if b, ferr := websense.Fetch(u, 30*time.Second); ferr == nil {
				txt = clipForContext(b.Text, 1200)
			}
		}
		evs = append(evs, ev{
			Title:   strings.TrimSpace(results[i].Title),
			URL:     u,
			Snippet: clipForContext(results[i].Snippet, 240),
			Body:    txt,
		})
	}
	evJSON, _ := json.MarshalIndent(evs, "", "  ")

	// Use scout model to extract structured interpretations.
	scoutModel := eg.ModelFor("scout", eg.ModelFor("speaker", "llama3.1:8b"))
	sys := `Du bist Bunny-Axiom-Extractor.
Aufgabe: Aus EVIDENCE extrahiere 2-6 konkrete, operationalisierbare Interpretationen für das Axiom.
Gib NUR JSON aus: {"items":[{"axiom_id":int,"kind":"definition|metric|rule|example|anti_example","key":"...","value":"...","confidence":0..1,"source_note":"domain/title"}]}
Regeln:
- Sei konkret, nicht philosophisch-vage.
- "metric": messbare Proxy-Signale (z.B. spam_rate, hallucination_risk, evidence_ratio).
- "rule": Konflikt-/Abwägungsregel (A1>A2>A3>A4 beibehalten; aber konkretisieren was "Schaden" bedeutet).
- confidence konservativ.`
	user := "AXIOM_ID: " + strconv.Itoa(ax.ID) + "\nAXIOM_TEXT: " + ax.Text + "\nEVIDENCE:\n" + string(evJSON)
	out, err := oc.Chat(scoutModel, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
	if err != nil {
		return nil
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return nil
	}
	out = stripCodeFenceIfAny(out)
	var parsed struct {
		Items []axiomItem `json:"items"`
	}
	if json.Unmarshal([]byte(out), &parsed) != nil || len(parsed.Items) == 0 {
		return nil
	}

	// Persist interpretations + commit metabolic cost/log.
	wrote := 0
	for _, it := range parsed.Items {
		if it.AxiomID < 1 || it.AxiomID > 4 {
			it.AxiomID = ax.ID
		}
		it.Kind = strings.TrimSpace(it.Kind)
		it.Key = strings.TrimSpace(it.Key)
		it.Value = strings.TrimSpace(it.Value)
		if it.Kind == "" || it.Key == "" || it.Value == "" {
			continue
		}
		if it.Confidence < 0 {
			it.Confidence = 0
		}
		if it.Confidence > 1 {
			it.Confidence = 1
		}
		if err := UpsertAxiomInterpretation(db, it.AxiomID, it.Kind, it.Key, it.Value, it.Confidence, it.SourceNote); err == nil {
			wrote++
		}
	}
	if wrote == 0 {
		return nil
	}

	// Self-change commit (metabolic brake + transparency log)
	ch := SelfChange{
		Kind:      "axiom",
		Target:    "axiom_interpretations",
		DeltaJSON: fmt.Sprintf(`{"axiom_id":%d,"items":%d}`, ax.ID, wrote),
		AxiomGoal: ax.ID,
		Risk:      RiskLow,
		Note:      "autonomous axiom enrichment via websense+scout",
	}
	CommitSelfChange(db, eg, body, ws, ch)
	return nil
}

func stripCodeFenceIfAny(s string) string {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "```") {
		return s
	}
	// remove first fence line
	lines := strings.Split(s, "\n")
	if len(lines) <= 2 {
		return strings.TrimSpace(strings.Trim(s, "`"))
	}
	lines = lines[1:]
	// drop last fence
	if strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

func kvInt(db *sql.DB, key string, fb int) int {
	if db == nil {
		return fb
	}
	var raw string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, strings.TrimSpace(key)).Scan(&raw)
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fb
	}
	n, err := strconv.Atoi(raw)
	if err != nil {
		return fb
	}
	return n
}

func kvSetInt(db *sql.DB, key string, v int) {
	if db == nil {
		return
	}
	_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?)
		ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		strings.TrimSpace(key), fmt.Sprintf("%d", v), time.Now().Format(time.RFC3339))
}
