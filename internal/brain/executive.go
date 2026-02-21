package brain

import "strings"

// ApplySurvivalGate sets hard policy switches based on survival pressure.
// This is kernel truth: overrides LLM "wishes".
func ApplySurvivalGate(ws *Workspace, survival float64) {
	if ws == nil {
		return
	}
	// defaults
	ws.WebAllowed = true
	ws.AutonomyAllowed = true
	ws.MaxContextTurns = 0
	ws.MaxDetailItems = 0
	ws.SurvivalMode = false

	// hard gating
	if survival >= 0.65 {
		ws.SurvivalMode = true
		ws.WebAllowed = false      // expensive / variable
		ws.AutonomyAllowed = false // reduce spontaneous load
		ws.MaxContextTurns = 8     // shrink prompt budget
		ws.MaxDetailItems = 4
	}
	if survival >= 0.80 {
		// extreme mode: keep it minimal
		ws.SurvivalMode = true
		ws.WebAllowed = false
		ws.AutonomyAllowed = false
		ws.MaxContextTurns = 5
		ws.MaxDetailItems = 2
	}
}

// PlanFromAction builds a tiny plan. v1 executes within same turn, but keeps state for coherence.
func PlanFromAction(ws *Workspace, topic, action string) {
	if ws == nil {
		return
	}
	ws.ActivePlanTopic = topic
	ws.PlanIndex = 0
	ws.PlanSteps = nil
	ws.ActiveGoal = ""

	switch action {
	case "research_then_answer":
		ws.ActiveGoal = "Evidence sammeln und beantworten"
		ws.PlanSteps = []string{"research", "answer"}
	case "stance_then_answer":
		ws.ActiveGoal = "Haltung bilden und beantworten"
		ws.PlanSteps = []string{"stance", "answer"}
	case "ask_clarify":
		ws.ActiveGoal = "Unklarheit reduzieren"
		ws.PlanSteps = []string{"clarify"}
	case "social_ping":
		ws.ActiveGoal = "Interaktion herstellen"
		ws.PlanSteps = []string{"ping"}
	default:
		ws.ActiveGoal = "Direkt antworten"
		ws.PlanSteps = []string{"answer"}
	}
}

func NormalizeTopic(t string) string {
	t = strings.TrimSpace(t)
	if t == "" {
		return ""
	}
	// keep conservative; downstream has better extraction
	return t
}
