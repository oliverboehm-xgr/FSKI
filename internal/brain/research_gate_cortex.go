package brain

import (
	"database/sql"
	"strings"

	"frankenstein-v0/internal/ollama"
)

// DecideResearchCortex is the cortex-level decision: when to use the web "sense".
// Design goal: prevent hallucinations (prefer false-positives over false-negatives).
//
// Composition:
//  1. existing heuristic gate (DecideResearch)
//  2. minimal hard guardrails (URL/link/time-stamp requests)
//  3. lightweight LLM gate (optional) that decides need_web under uncertainty
//
// Important: In training dry-runs we must avoid sensor calls.
func DecideResearchCortex(db *sql.DB, oc *ollama.Client, gateModel string, userText string, intent Intent, ws *Workspace, tr *Traits, dr *Drives, aff *AffectState) ResearchDecision {
	// Never call sensors in dry runs.
	if ws != nil && ws.TrainingDryRun {
		return ResearchDecision{}
	}

	base := DecideResearch(db, userText, intent, ws, tr, dr, aff)

	// If web is blocked, never request web.
	if ws != nil && !ws.WebAllowed {
		base.Do = false
		base.Score = 0
		if base.Reason == "" {
			base.Reason = "web_blocked"
		} else {
			base.Reason = base.Reason + ",web_blocked"
		}
		return base
	}

	// Minimal guardrail triggers (cheap, generic).
	if ok, why := HardEvidenceTrigger(userText); ok {
		base.Do = true
		if base.Score < 0.90 {
			base.Score = 0.90
		}
		base.Reason = appendReason(base.Reason, why)
	}

	// LLM gate: if still not requesting research, ask a small model whether we need web.
	// Policy: if uncertain => need_web=true.
	if !base.Do {
		need, conf, q, why, err := CortexWebGate(oc, gateModel, userText, intent, ws)
		if err == nil {
			if need {
				base.Do = true
			}
			if conf > base.Score {
				base.Score = conf
			}
			if strings.TrimSpace(q) != "" {
				base.Query = NormalizeSearchQuery(q)
			}
			base.Reason = appendReason(base.Reason, why)
		}
	}

	// Safety: EXTERNAL_FACT should not be answered without evidence.
	if intent == IntentExternalFact && !base.Do {
		base.Do = true
		if base.Score < 0.75 {
			base.Score = 0.75
		}
		base.Reason = appendReason(base.Reason, "external_fact_safety")
	}

	// Ensure query exists when research is requested.
	if base.Do && strings.TrimSpace(base.Query) == "" {
		base.Query = NormalizeSearchQuery(userText)
	}

	return base
}

func appendReason(existing, add string) string {
	add = strings.TrimSpace(add)
	if add == "" {
		return strings.TrimSpace(existing)
	}
	existing = strings.TrimSpace(existing)
	if existing == "" {
		return add
	}
	return existing + "," + add
}
