package brain

import (
	"regexp"
	"strings"

	"frankenstein-v0/internal/epi"
)

type Intent int

const (
	IntentUnknown Intent = iota
	IntentMetaBunny
	IntentUserLife
	IntentTaskTech
	IntentExternalFact
	IntentOpinion
	IntentResearchCommand
)

// DetectIntent is now data-driven (epigenetic). If eg==nil or no rules, it falls back to UNKNOWN.
func DetectIntentWithEpigenome(s string, eg *epi.Epigenome) Intent {
	t := normalizeText(s)
	if eg == nil {
		return IntentUnknown
	}
	rules := eg.IntentRules()
	if len(rules) == 0 {
		return IntentUnknown
	}
	for _, r := range rules {
		if matchRule(t, r) {
			return mapIntentString(r.Intent)
		}
	}
	return IntentUnknown
}

func matchRule(t string, r epi.IntentRule) bool {
	for _, c := range r.Contains {
		if c == "" {
			continue
		}
		if strings.Contains(t, strings.ToLower(c)) {
			return true
		}
	}
	for _, pat := range r.Regex {
		re, err := regexp.Compile(pat)
		if err != nil {
			continue
		}
		if re.MatchString(t) {
			return true
		}
	}
	return false
}

func mapIntentString(s string) Intent {
	switch strings.ToUpper(strings.TrimSpace(s)) {
	case "META_BUNNY":
		return IntentMetaBunny
	case "EXTERNAL_FACT":
		return IntentExternalFact
	case "OPINION":
		return IntentOpinion
	case "RESEARCH_CMD":
		return IntentResearchCommand
	case "USER_LIFE":
		return IntentUserLife
	case "TASK_TECH":
		return IntentTaskTech
	default:
		return IntentUnknown
	}
}

func IntentToMode(i Intent) string {
	switch i {
	case IntentMetaBunny:
		return "META_BUNNY"
	case IntentExternalFact:
		return "EXTERNAL_FACT"
	case IntentOpinion:
		return "OPINION"
	case IntentResearchCommand:
		return "RESEARCH_CMD"
	case IntentUserLife:
		return "USER_LIFE"
	case IntentTaskTech:
		return "TASK_TECH"
	default:
		return "GENERAL"
	}
}

func normalizeText(s string) string {
	t := strings.ToLower(strings.TrimSpace(s))
	t = strings.ReplaceAll(t, "\t", " ")
	for strings.Contains(t, "  ") {
		t = strings.ReplaceAll(t, "  ", " ")
	}
	return t
}
