package brain

import (
	"database/sql"
	"strings"

	"frankenstein-v0/internal/ollama"
)

// DecideResearchCortex is the cortex-level decision: when to use the web "sense".
// It combines:
// - tiny hard guardrails (links/urls)
// - the existing heuristic gate (DecideResearch)
// - an optional lightweight LLM gate (CortexWebGate)
//
// Policy: prefer false-positives over false-negatives (hallucinations are worse).
func DecideResearchCortex(db *sql.DB, oc *ollama.Client, gateModel string, userText string, intent Intent, ws *Workspace, tr *Traits, dr *Drives, aff *AffectState) ResearchDecision {
	// Training dry-run must avoid extra calls.
	if ws != nil && ws.TrainingDryRun {
		return ResearchDecision{}
	}

	base := DecideResearch(db, userText, intent, ws, tr, dr, aff)

	// If web is blocked, never request web.
	if ws != nil && !ws.WebAllowed {
		base.Do = false
		base.Reason = "web_blocked"
		base.Score = 0
		return base
	}

	// Minimal guardrail
	if ok, why := HardEvidenceTrigger(userText); ok {
		base.Do = true
		base.Score = 1.0
		if base.Reason != "" {
			base.Reason = why + "," + base.Reason
		} else {
			base.Reason = why
		}
	}

	// Lightweight LLM gate (prefers need_web=true under uncertainty)
	if !base.Do && ShouldConsiderWeb(userText, intent) {
		survivalMode := false
		if ws != nil {
			survivalMode = ws.SurvivalMode
		}
		need, conf, q, why, err := CortexWebGate(oc, gateModel, userText, intent, survivalMode)
		if err == nil {
			// combine: allow only escalation (never suppress a positive base decision)
			if need {
				base.Do = true
			}
			if conf > base.Score {
				base.Score = conf
			}
			if strings.TrimSpace(q) != "" {
				base.Query = NormalizeSearchQuery(q)
			}
			if strings.TrimSpace(why) != "" {
				if base.Reason != "" {
					base.Reason = base.Reason + "," + why
				} else {
					base.Reason = why
				}
			}
		}
	}

	// Safety: external facts should not be guessed; escalate if uncertain.
	if intent == IntentExternalFact && !base.Do {
		base.Do = true
		if base.Score < 0.75 {
			base.Score = 0.75
		}
		if base.Reason != "" {
			base.Reason = base.Reason + ",external_fact_safety"
		} else {
			base.Reason = "external_fact_safety"
		}
	}

	// Ensure query exists when research is requested.
	if base.Do {
		if strings.TrimSpace(base.Query) == "" {
			base.Query = NormalizeSearchQuery(userText)
		}
	}

	return base
}
