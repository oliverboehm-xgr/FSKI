package brain

import (
	"database/sql"
	"strings"
)

type ResearchDecision struct {
	Do     bool
	Query  string
	Reason string
	Score  float64
}

// DecideResearch is the generic kernel gate:
// decide when to use "senses" (web) to make progress.
// It is topic-agnostic. It uses:
// - explicit user request ("recherchiere", "im internet")
// - intent (external fact / opinion / unknown)
// - concept confidence (if known)
// - drives/traits/affects as modulation
func DecideResearch(db *sql.DB, userText string, intent Intent, ws *Workspace, tr *Traits, dr *Drives, aff *AffectState) ResearchDecision {
	t := strings.ToLower(strings.TrimSpace(userText))
	if t == "" {
		return ResearchDecision{}
	}

	// Base score
	score := 0.0
	reason := []string{}

	// Explicit request
	if isResearchLike(t) {
		score += 0.95
		reason = append(reason, "user_requested_research")
	}

	// Intents that usually require evidence to avoid BS
	switch intent {
	case IntentExternalFact:
		score += 0.80
		reason = append(reason, "external_fact")
	case IntentOpinion:
		// opinion can be grounded in sources; if user asks "äußere dich", prefer evidence
		score += 0.55
		reason = append(reason, "opinion_requested")
	}

	// Unknown concept / low confidence => research helps
	term, _ := ExtractCandidate(userText)
	if term != "" {
		c, ok := GetConcept(db, term)
		if !ok {
			score += 0.55
			reason = append(reason, "unknown_concept")
		} else if c.Confidence < 0.45 {
			score += 0.35
			reason = append(reason, "low_concept_confidence")
		}
	}

	// Modulators
	// curiosity & research_bias raise score; negative affects inhibit
	cur := 0.45
	if dr != nil {
		cur = dr.Curiosity
	}
	rb := 0.55
	if tr != nil {
		rb = tr.ResearchBias
	}
	score += 0.25 * clamp01(cur)
	score += 0.35 * clamp01(rb)

	inhib := 0.0
	if aff != nil {
		inhib += 0.8 * aff.Get("shame")
		inhib += 0.4 * aff.Get("unwell")
		inhib += 0.3 * aff.Get("pain")
		inhib += 0.3 * aff.Get("fear")
	}
	score -= 0.35 * clamp01(inhib)

	// Avoid research for pure meta-self questions unless explicitly requested
	if intent == IntentMetaBunny && !isResearchLike(t) {
		score = 0
		reason = []string{"meta_self"}
	}

	score = clamp01(score)

	// Threshold: adaptive baseline (higher research_bias lowers threshold)
	thr := 0.70 - 0.20*clamp01(rb) // rb=1 => thr 0.50, rb=0 => thr 0.70
	if thr < 0.45 {
		thr = 0.45
	}

	do := score >= thr

	// Query selection:
	// - if the text is just "recherchiere" or similar -> use previous user turn if available
	query := userText
	if isBareResearchCommand(t) && ws != nil && ws.PrevUserText != "" {
		query = ws.PrevUserText
	}
	query = NormalizeSearchQuery(query)

	return ResearchDecision{
		Do:     do,
		Query:  query,
		Reason: strings.Join(reason, ","),
		Score:  score,
	}
}

func isResearchLike(t string) bool {
	return strings.Contains(t, "recherch") ||
		strings.Contains(t, "im internet") ||
		strings.Contains(t, "nachsehen") ||
		strings.Contains(t, "schau nach") ||
		strings.Contains(t, "quelle") ||
		strings.Contains(t, "quellen") ||
		strings.Contains(t, "link") ||
		strings.Contains(t, "url") ||
		strings.Contains(t, "nachricht") ||
		strings.Contains(t, "news") ||
		looksLikeURLOrDomain(t)
}

func isBareResearchCommand(t string) bool {
	// minimal: "recherchiere", "schau nach", "im internet nachsehen"
	// i.e. a command without subject/topic
	short := strings.ReplaceAll(t, "?", "")
	short = strings.TrimSpace(short)
	if len(short) <= 18 && (strings.Contains(short, "recherch") || strings.Contains(short, "nachsehen")) {
		return true
	}
	if short == "im internet" || short == "im internet nachsehen" || short == "schau nach" {
		return true
	}
	return false
}

func looksLikeURLOrDomain(t string) bool {
	if strings.Contains(t, "http://") || strings.Contains(t, "https://") || strings.Contains(t, "www.") {
		return true
	}
	// crude TLD hint (good enough for gating)
	for _, tld := range []string{".de", ".com", ".org", ".net", ".io", ".eu"} {
		if strings.Contains(t, tld) {
			return true
		}
	}
	return false
}
