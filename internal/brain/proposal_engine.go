package brain

import (
	"database/sql"
	"encoding/json"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
)

type ProposalIdea struct {
	Kind  string `json:"kind"`
	Title string `json:"title"`
	Body  string `json:"body"`
	Note  string `json:"note"`
}

func frictionScore(db *sql.DB, aff *AffectState) float64 {
	s := 0.0
	if aff != nil {
		s += 0.7*aff.Get("shame") + 0.3*aff.Get("pain")
	}
	if db != nil {
		var n int
		_ = db.QueryRow(`SELECT COUNT(*) FROM caught_events WHERE created_at >= ?`, time.Now().Add(-30*time.Minute).Format(time.RFC3339)).Scan(&n)
		s += 0.15 * float64(n)
	}
	return clamp01(s)
}

func tooManyProposalsThisHour(db *sql.DB, maxPerHour int) bool {
	if db == nil {
		return true
	}
	if maxPerHour <= 0 {
		return true
	}
	var n int
	_ = db.QueryRow(`SELECT COUNT(*) FROM thought_proposals WHERE created_at >= ?`, time.Now().Add(-1*time.Hour).Format(time.RFC3339)).Scan(&n)
	return n >= maxPerHour
}

func lastProposalAt(db *sql.DB) time.Time {
	if db == nil {
		return time.Time{}
	}
	var ts string
	_ = db.QueryRow(`SELECT created_at FROM thought_proposals ORDER BY id DESC LIMIT 1`).Scan(&ts)
	t, _ := time.Parse(time.RFC3339, ts)
	return t
}

func GenerateProposalIdeas(db *sql.DB, ws *Workspace, aff *AffectState) []ProposalIdea {
	var out []ProposalIdea
	if ws == nil {
		return out
	}
	hint := strings.ToLower(strings.TrimSpace(ws.CurrentThought + "\n" + ws.InnerSpeech))
	if strings.Contains(hint, "ollama") || strings.Contains(hint, "llm") {
		out = append(out, ProposalIdea{Kind: "code", Title: "LLM health guard / auto-start", Body: "Add Ollama ping + graceful fallback + optional auto-start/pull.", Note: "derived from thought text"})
	}
	if strings.Contains(hint, "topic") || strings.Contains(hint, "drift") || strings.Contains(hint, "nochmal") {
		out = append(out, ProposalIdea{Kind: "code", Title: "Topic drift fix", Body: "Replace whitelist topic regex with open-vocabulary + info-gate anchor; prevent lock-in.", Note: "derived from thought text"})
	}
	if frictionScore(db, aff) >= 0.6 {
		out = append(out, ProposalIdea{Kind: "epigenetic", Title: "Reduce clarify loop bias", Body: `{"policy":"penalize_ask_clarify","delta":-0.2}`, Note: "friction high; reduce loops"})
	}
	return out
}

func SaveThoughtProposal(db *sql.DB, idea ProposalIdea) (int64, error) {
	if db == nil {
		return 0, nil
	}
	now := time.Now().Format(time.RFC3339)
	payload := strings.TrimSpace(idea.Body)
	if payload == "" {
		payload = "{}"
	}
	res, err := db.Exec(`INSERT INTO thought_proposals(created_at,kind,title,payload,status,note) VALUES(?,?,?,?,?,?)`,
		now, idea.Kind, idea.Title, payload, "proposed", idea.Note)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func TickProposalEngine(db *sql.DB, eg *epi.Epigenome, ws *Workspace, aff *AffectState) (created int, msg string) {
	if db == nil || eg == nil || ws == nil {
		return 0, ""
	}
	enabled, minInt, maxPerHour, frTh, _ := eg.ProposalEngineParams()
	if !enabled || tooManyProposalsThisHour(db, maxPerHour) {
		return 0, ""
	}
	last := lastProposalAt(db)
	if !last.IsZero() && time.Since(last).Seconds() < minInt {
		return 0, ""
	}
	if frictionScore(db, aff) < frTh && strings.TrimSpace(ws.InnerSpeech) == "" && strings.TrimSpace(ws.CurrentThought) == "" {
		return 0, ""
	}
	ideas := GenerateProposalIdeas(db, ws, aff)
	for _, it := range ideas {
		if _, err := SaveThoughtProposal(db, it); err == nil {
			created++
		}
	}
	if created > 0 {
		b, _ := json.Marshal(map[string]any{"created": created})
		return created, "Ich habe " + strconv.Itoa(created) + " Selbstverbesserungs-Vorschläge aus meiner Gedankenwelt erzeugt. (thought_proposals) " + string(b)
	}
	return 0, ""
}
