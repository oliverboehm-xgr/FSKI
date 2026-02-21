package main

import (
	"database/sql"
	"encoding/json"
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
