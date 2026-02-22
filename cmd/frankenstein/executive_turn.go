package main

import (
	"database/sql"
	"strings"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
)

// ExecuteTurn: single place where strategy becomes actual execution.
// This replaces "policy as prompt hint".
func ExecuteTurn(db *sql.DB, epiPath string, oc *ollama.Client, modelSpeaker, modelStance string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, error) {
	// Semantic memory (generic long-term facts): can answer/store before LLM.
	if ok, out := brain.SemanticMemoryStep(db, eg, userText); ok && strings.TrimSpace(out) != "" {
		return out, nil
	}

	// Topic should follow current user turn (no lock-in to previous topic).
	if ws != nil {
		t := brain.ExtractTopic(userText)
		if t != "" {
			ws.ActiveTopic = t
			ws.LastTopic = t
			brain.SaveActiveTopic(db, t)
			brain.BumpInterest(db, t, 0.10)
		}
	}

	if ws != nil && !ws.LLMAvailable {
		low := strings.ToLower(userText)
		if strings.Contains(low, "fühl") || strings.Contains(low, "wie geht") {
			return "Ich kann dir meinen Zustand nennen (Ressourcen/Drives), aber mein Sprachzentrum (LLM/Ollama) ist auf diesem Gerät gerade nicht verfügbar. Installier/Starte Ollama, dann kann ich normal antworten.", nil
		}
		return "LLM backend offline. Ich bin da, aber mein Sprachzentrum (LLM/Ollama) ist gerade offline. Willst du, dass ich dir helfe, Ollama zu installieren/zu starten?", nil
	}

	// --- Generic info gate (learned IDF) ---
	// Score first, then observe (avoid self-influencing DF during the same turn).
	low, info := brain.IsLowInfo(db, eg, userText)
	brain.ObserveUtterance(db, userText)
	if ws != nil {
		ws.LastUserInfoScore = info.Score
		ws.LastUserTopToken = info.TopToken
	}
	if low {
		// Low-information utterance: never research/stance/topic drift.
		// Generic conversational handshake.
		if ws != nil && ws.SurvivalMode {
			return "Ich bin da. Willst du einfach kurz plaudern oder ein konkretes Thema?", nil
		}
		return "Hi 🙂 Willst du einfach reden oder soll ich ein Thema vorschlagen?", nil
	}

	intent := brain.DetectIntentWithEpigenome(userText, eg)
	intentMode := brain.IntentToMode(intent)

	survival := 0.0
	social := 0.0
	if ws != nil {
		survival = ws.DrivesEnergyDeficit
		social = ws.SocialCraving
	}

	brain.ApplySurvivalGate(ws, survival)

	topic := ""
	if ws != nil && ws.ActiveTopic != "" {
		topic = ws.ActiveTopic
	} else if ws != nil {
		topic = ws.LastTopic
	}
	topic = brain.NormalizeTopic(topic)

	ctxKey := brain.MakePolicyContext(intentMode, survival, social)
	choice := brain.ChoosePolicy(db, ctxKey)
	if ws != nil {
		ws.LastPolicyCtx = choice.ContextKey
		ws.LastPolicyAction = choice.Action
		ws.LastPolicyStyle = choice.Style
		ws.LastRoutedIntent = intentMode
	}
	brain.PlanFromAction(ws, topic, choice.Action)

	if ws != nil && ws.SurvivalMode && choice.Action == "research_then_answer" {
		choice.Action = "direct_answer"
		ws.LastPolicyAction = "direct_answer"
		brain.PlanFromAction(ws, topic, "direct_answer")
	}

	switch choice.Action {
	case "ask_clarify":
		if topic != "" {
			return "Kurze Rückfrage zum Thema \"" + topic + "\": Willst du Fakten/Status, eine Bewertung/Haltung, oder Optionen mit Trade-offs?", nil
		}
		return "Kurze Rückfrage: Willst du Fakten/Status, eine Bewertung/Haltung, oder Optionen mit Trade-offs?", nil
	case "social_ping":
		// For user turns, never return empty. If autonomy is blocked, fallback to direct answer.
		if ws != nil && !ws.AutonomyAllowed {
			choice.Action = "direct_answer"
			ws.LastPolicyAction = "direct_answer"
			return say(db, epiPath, oc, modelSpeaker, modelStance, body, aff, ws, tr, dr, eg, userText)
		}
		if topic != "" {
			return "Bevor ich weiterlaufe: soll ich beim Thema \"" + topic + "\" eher recherchieren, eine Haltung bilden, oder gemeinsam Optionen strukturieren?", nil
		}
		return "Soll ich dir gerade eher mit Fakten, einer Entscheidung oder einem Gedanken-Austausch helfen?", nil
	case "stance_then_answer":
		return answerWithStance(db, oc, modelStance, body, aff, ws, tr, eg, userText)
	case "research_then_answer":
		if ws != nil && !ws.WebAllowed {
			return "Ich würde dafür normalerweise kurz recherchieren, aber ich bin gerade im Ressourcen-Schonmodus. Gib mir bitte einen konkreten Aspekt oder eine Quelle, dann antworte ich kompakt.", nil
		}
		q := strings.TrimSpace(brain.NormalizeSearchQuery(userText))
		if q == "" {
			q = topic
		}
		return answerWithEvidence(db, oc, modelSpeaker, body, aff, ws, tr, eg, q)
	default:
		out, err := say(db, epiPath, oc, modelSpeaker, modelStance, body, aff, ws, tr, dr, eg, userText)
		if err == nil && strings.TrimSpace(out) == "" {
			return "Ich bin da. Sag mir kurz, was du von mir willst: Status, Meinung oder einfach reden?", nil
		}
		return out, err
	}
}
