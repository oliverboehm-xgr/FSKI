package main

import (
	"database/sql"
	"strings"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
)

type MutantOverlay struct {
	Strength float64
	Prompt   string
	Model    string
}

func ExecuteTurnWithMeta(db *sql.DB, epiPath string, oc *ollama.Client, modelSpeaker, modelStance string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string, mut *MutantOverlay) (out string, action string, style string, ctxKey string, topic string, intentMode string) {
	nb := brain.NewNBIntent(db)
	intent := brain.DetectIntentHybrid(userText, eg, nb)
	// Research gate (truth kernel)
	rd := brain.DecideResearch(db, userText, intent, ws, tr, dr, aff)
	intentMode = brain.IntentToMode(intent)
	topic = ws.ActiveTopic
	if topic == "" {
		topic = ws.LastTopic
	}
	ctxKey = brain.MakePolicyContext(intentMode, ws.DrivesEnergyDeficit, ws.SocialCraving)
	choice := brain.ChoosePolicy(db, ctxKey)
	action = choice.Action
	style = choice.Style
	if ws != nil {
		ws.LastPolicyCtx = choice.ContextKey
		ws.LastPolicyAction = action
		ws.LastPolicyStyle = style
		ws.LastRoutedIntent = intentMode
	}

	speakerModel := modelSpeaker
	extraPrompt := ""
	if mut != nil {
		if strings.TrimSpace(mut.Model) != "" {
			speakerModel = mut.Model
		}
		extraPrompt = strings.TrimSpace(mut.Prompt)
		if action == "ask_clarify" && mut.Strength >= 0.15 {
			action = "direct_answer"
		}
		if mut.Strength >= 0.25 {
			style = "direct"
		}
	}

	// Truth-gate: if research is indicated, avoid direct_answer bluffing.
	if rd.Do && ws != nil && ws.WebAllowed && action == "direct_answer" {
		action = "research_then_answer"
		ws.LastPolicyAction = action
	}

	switch action {
	case "ask_clarify":
		return "Kurze Rückfrage: Willst du Fakten/Status, eine Bewertung/Haltung, oder Optionen mit Trade-offs?", action, style, ctxKey, topic, intentMode
	case "research_then_answer":
		q := brain.NormalizeSearchQuery(userText)
		if rd.Do && strings.TrimSpace(rd.Query) != "" {
			q = rd.Query
		}
		out2, err := answerWithEvidence(db, oc, speakerModel, body, aff, ws, tr, eg, q)
		if err != nil {
			return "Fehler bei Recherche/Antwort (LLM/Web).", action, style, ctxKey, topic, intentMode
		}
		return out2, action, style, ctxKey, topic, intentMode
	case "stance_then_answer":
		out2, err := answerWithStance(db, oc, modelStance, body, aff, ws, tr, eg, userText)
		if err != nil {
			return "Fehler bei Haltung/Antwort (LLM).", action, style, ctxKey, topic, intentMode
		}
		return out2, action, style, ctxKey, topic, intentMode
	default:
		out2, err := sayWithMutation(db, epiPath, oc, speakerModel, modelStance, body, aff, ws, tr, dr, eg, userText, extraPrompt)
		if err != nil {
			return "Fehler beim Antworten (LLM).", action, style, ctxKey, topic, intentMode
		}
		return out2, action, style, ctxKey, topic, intentMode
	}
}
