package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

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

var reNumberRef = regexp.MustCompile(`(?i)\b(?:nachricht|punkt|nr\.?|nummer|article|artikel)\s*#?\s*([1-9][0-9]?)\b`)

func BuildFollowupAnchor(db *sql.DB, userText string) string {
	if db == nil {
		return ""
	}
	m := reNumberRef.FindStringSubmatch(userText)
	if len(m) < 2 {
		return ""
	}
	n, err := strconv.Atoi(m[1])
	if err != nil || n <= 0 {
		return ""
	}

	var prevReply string
	err = db.QueryRow(
		`SELECT m.text
		 FROM messages m
		 LEFT JOIN message_meta mm ON mm.message_id = m.id
		 WHERE COALESCE(mm.kind,'reply')='reply'
		 ORDER BY m.id DESC
		 LIMIT 1`,
	).Scan(&prevReply)
	if err != nil {
		return ""
	}
	item := ExtractNumberedListItem(prevReply, n)
	if item == "" {
		return ""
	}
	return fmt.Sprintf("FOLLOWUP_ANCHOR: User referenziert Punkt %d aus der letzten Bunny-Antwort.\nPUNKT_%d: %s", n, n, item)
}

func ExtractNumberedListItem(text string, n int) string {
	if n <= 0 {
		return ""
	}
	lines := strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n")
	prefixA := strconv.Itoa(n) + "."
	prefixB := "**" + strconv.Itoa(n) + "."
	for _, line := range lines {
		s := strings.TrimSpace(line)
		if s == "" {
			continue
		}
		if strings.HasPrefix(s, prefixA) {
			s = strings.TrimSpace(strings.TrimPrefix(s, prefixA))
			return strings.Trim(s, " *")
		}
		if strings.HasPrefix(s, prefixB) {
			s = strings.TrimSpace(strings.TrimPrefix(s, prefixB))
			s = strings.TrimPrefix(s, "**")
			return strings.Trim(s, " *")
		}
	}
	return ""
}
