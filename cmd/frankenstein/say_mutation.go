package main

import (
	"database/sql"
	"strings"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
)

func sayWithMutation(db *sql.DB, epiPath string, oc *ollama.Client, model string, stanceModel string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string, mutantPrompt string) (string, error) {
	if strings.TrimSpace(mutantPrompt) == "" {
		return say(db, epiPath, oc, model, stanceModel, body, aff, ws, tr, dr, eg, userText)
	}
	old := ""
	if ws != nil {
		old = ws.InnerSpeech
		ws.InnerSpeech = strings.TrimSpace(mutantPrompt) + "\n" + old
	}
	out, err := say(db, epiPath, oc, model, stanceModel, body, aff, ws, tr, dr, eg, userText)
	if ws != nil {
		ws.InnerSpeech = old
	}
	return out, err
}
