package main

import (
	"database/sql"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
	"frankenstein-v0/internal/websense"
)

// ExecuteTurn: single place where strategy becomes actual execution.
// This replaces "policy as prompt hint".
func ExecuteTurn(db *sql.DB, epiPath string, oc *ollama.Client, modelSpeaker, modelStance string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, error) {
	// UI commands (were previously only available in console loop).
	if ok, out := handleWebCommands(userText); ok {
		return out, nil
	}
	if ok, out := handleThoughtCommands(db, userText); ok {
		return out, nil
	}
	// Natural confirmation: if last auto asked about materializing thought_proposals and user says "ja", show them.
	if isAffirmative(userText) && brain.CountThoughtProposals(db, "proposed") > 0 && lastAutoAsked(db, "ausformulieren", 10*time.Minute) {
		return brain.RenderThoughtProposalList(db, 10), nil
	}

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

func isAffirmative(s string) bool {
	t := strings.TrimSpace(strings.ToLower(s))
	switch t {
	case "ja", "j", "jo", "yes", "y", "ok", "okay", "klar", "bitte", "gerne", "mach", "mach das", "mach es", "genau":
		return true
	default:
		return false
	}
}

func lastAutoAsked(db *sql.DB, contains string, within time.Duration) bool {
	if db == nil {
		return false
	}
	var txt string
	var ts string
	_ = db.QueryRow(`SELECT m.text, m.created_at
		FROM messages m
		JOIN message_meta mm ON mm.message_id=m.id
		WHERE mm.kind='auto'
		ORDER BY m.id DESC LIMIT 1`).Scan(&txt, &ts)
	txt = strings.ToLower(strings.TrimSpace(txt))
	if txt == "" || !strings.Contains(txt, strings.ToLower(contains)) {
		return false
	}
	tm, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return false
	}
	return time.Since(tm) <= within
}

func handleWebCommands(userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/web") && !strings.HasPrefix(line, "/websense") {
		return false, ""
	}
	// Usage:
	// /web test <query>
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return true, "Use: /web test <query>"
	}
	if parts[1] != "test" {
		return true, "Use: /web test <query>"
	}
	q := strings.TrimSpace(strings.TrimPrefix(line, parts[0]+" "+parts[1]))
	q = strings.TrimSpace(q)
	if q == "" {
		return true, "Use: /web test <query>"
	}
	results, err := websense.Search(q, 6)
	if err != nil || len(results) == 0 {
		if err != nil {
			return true, "websense.Search failed: " + err.Error()
		}
		return true, "Keine Ergebnisse."
	}
	var b strings.Builder
	b.WriteString("websense.Search OK. Top Ergebnisse:\n")
	for i := 0; i < len(results) && i < 5; i++ {
		title := strings.TrimSpace(results[i].Title)
		u := strings.TrimSpace(results[i].URL)
		sn := strings.TrimSpace(results[i].Snippet)
		if len(sn) > 140 {
			sn = sn[:140] + "..."
		}
		b.WriteString("- " + title + "\n  " + u + "\n  " + sn + "\n")
	}
	return true, strings.TrimSpace(b.String())
}
func handleThoughtCommands(db *sql.DB, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/thought") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) == 1 {
		return true, brain.RenderThoughtProposalList(db, 10)
	}
	switch parts[1] {
	case "list":
		return true, brain.RenderThoughtProposalList(db, 10)
	case "show":
		if len(parts) < 3 {
			return true, "Use: /thought show <id>"
		}
		id, _ := strconv.ParseInt(parts[2], 10, 64)
		return true, brain.RenderThoughtProposal(db, id)
	case "materialize":
		if len(parts) < 3 {
			return true, "Use: /thought materialize <id|all>"
		}
		arg := strings.ToLower(strings.TrimSpace(parts[2]))
		if arg == "all" {
			return true, brain.MaterializeAllThoughtProposals(db, 25)
		}
		id, _ := strconv.ParseInt(arg, 10, 64)
		msg, _ := brain.MaterializeThoughtProposal(db, id)
		return true, msg
	default:
		return true, "Use: /thought list | /thought show <id> | /thought materialize <id|all>"
	}
}
