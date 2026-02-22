package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"unicode"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
)

// BuildHumanContext assembles gist+details+turns, respecting SurvivalGate overrides from Workspace.
func BuildHumanContext(db *sql.DB, eg *epi.Epigenome, ws *brain.Workspace) (activeTopic, gist, details, concepts, stance, turns string) {
	if ws != nil {
		activeTopic = strings.TrimSpace(ws.ActiveTopic)
	}
	if activeTopic == "" && ws != nil {
		activeTopic = strings.TrimSpace(ws.LastTopic)
	}

	// memory params with overrides
	_, ctxTurns, detItems, _, _, _, _ := eg.MemoryParams()
	if ws != nil {
		if ws.MaxContextTurns > 0 {
			ctxTurns = ws.MaxContextTurns
		}
		if ws.MaxDetailItems > 0 {
			detItems = ws.MaxDetailItems
		}
	}
	turns = brain.BuildDialogContext(db, ctxTurns)

	if activeTopic != "" {
		if s, ok := brain.GetLastEpisode(db, activeTopic); ok {
			gist = s
		}
		if detItems > 0 {
			details = brain.RecallDetails(db, activeTopic, detItems)
		}
		concepts = brain.RecallConcepts(db, activeTopic, 4)
		if st, ok := brain.GetStance(db, activeTopic); ok {
			stanceJSON, _ := json.MarshalIndent(map[string]any{
				"label": st.Label, "position": st.Position, "confidence": st.Confidence, "rationale": st.Rationale,
			}, "", "  ")
			stance = string(stanceJSON)
		}
	}
	return
}

type refCandidate struct {
	MessageID int64
	Kind      string
	Text      string
	Score     float64
}

var reToken = regexp.MustCompile(`(?i)[\p{L}\p{N}]{2,}`)

func BuildReferenceCandidates(db *sql.DB, userText string, limit int) string {
	if db == nil || strings.TrimSpace(userText) == "" || limit <= 0 {
		return ""
	}
	rows, err := db.Query(
		`SELECT m.id, COALESCE(mm.kind,'reply') AS kind, m.text
		 FROM messages m
		 LEFT JOIN message_meta mm ON mm.message_id = m.id
		 WHERE COALESCE(mm.kind,'reply') IN ('user','reply','auto')
		 ORDER BY m.id DESC
		 LIMIT 36`,
	)
	if err != nil {
		return ""
	}
	defer rows.Close()

	queryTokens := tokenSet(userText)
	if len(queryTokens) == 0 {
		return ""
	}

	var cands []refCandidate
	idx := 0
	for rows.Next() {
		var id int64
		var kind, text string
		if err := rows.Scan(&id, &kind, &text); err != nil {
			continue
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		score := scoreCandidate(queryTokens, userText, text, idx)
		idx++
		if score <= 0.16 {
			continue
		}
		cands = append(cands, refCandidate{MessageID: id, Kind: kind, Text: clipLine(text, 220), Score: score})
	}
	if len(cands) == 0 {
		return ""
	}
	sort.Slice(cands, func(i, j int) bool {
		if cands[i].Score == cands[j].Score {
			return cands[i].MessageID > cands[j].MessageID
		}
		return cands[i].Score > cands[j].Score
	})
	if limit > len(cands) {
		limit = len(cands)
	}
	var b strings.Builder
	b.WriteString("REFERENCE_CANDIDATES (recency+similarity):\n")
	for i := 0; i < limit; i++ {
		c := cands[i]
		role := "Bunny"
		if c.Kind == "user" {
			role = "User"
		}
		fmt.Fprintf(&b, "- [%s#%d score=%.2f] %s\n", role, c.MessageID, c.Score, c.Text)
	}
	return strings.TrimSpace(b.String())
}

func scoreCandidate(queryTokens map[string]struct{}, userText, candText string, recencyIdx int) float64 {
	candTokens := tokenSet(candText)
	if len(candTokens) == 0 {
		return 0
	}
	overlap := 0
	for t := range queryTokens {
		if _, ok := candTokens[t]; ok {
			overlap++
		}
	}
	base := float64(overlap) / float64(len(queryTokens))
	if base == 0 {
		return 0
	}
	recency := 1.0 / (1.0 + 0.22*float64(recencyIdx))
	shortBoost := 1.0
	if len([]rune(strings.TrimSpace(userText))) <= 64 {
		shortBoost = 1.15
	}
	if hasContextReferenceCue(userText) {
		shortBoost += 0.15
	}
	return base * recency * shortBoost
}

func hasContextReferenceCue(s string) bool {
	t := strings.ToLower(strings.TrimSpace(s))
	cues := []string{"dazu", "darüber", "darueber", "davon", "oben", "vorhin", "letzte", "genannte", "nochmal", "dieser", "diese", "diesen", "die "}
	for _, c := range cues {
		if strings.Contains(t, c) {
			return true
		}
	}
	return false
}

func tokenSet(s string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, tok := range reToken.FindAllString(strings.ToLower(s), -1) {
		tok = strings.TrimSpace(tok)
		if len(tok) < 2 || isStopToken(tok) {
			continue
		}
		out[tok] = struct{}{}
	}
	return out
}

func isStopToken(tok string) bool {
	if tok == "" {
		return true
	}
	if len([]rune(tok)) <= 2 {
		return true
	}
	sw := map[string]struct{}{
		"und": {}, "oder": {}, "aber": {}, "dann": {}, "noch": {}, "eine": {}, "einer": {}, "eines": {}, "der": {}, "die": {}, "das": {}, "den": {},
		"mit": {}, "von": {}, "für": {}, "fuer": {}, "über": {}, "ueber": {}, "ist": {}, "sind": {}, "war": {}, "was": {}, "wie": {}, "bitte": {},
	}
	_, ok := sw[tok]
	return ok
}

func clipLine(s string, n int) string {
	s = strings.TrimSpace(strings.ReplaceAll(strings.ReplaceAll(s, "\r", " "), "\n", " "))
	s = strings.Map(func(r rune) rune {
		if unicode.IsSpace(r) {
			return ' '
		}
		return r
	}, s)
	s = strings.Join(strings.Fields(s), " ")
	if len([]rune(s)) <= n {
		return s
	}
	r := []rune(s)
	return string(r[:n]) + "…"
}
