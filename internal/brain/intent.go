package brain

import "strings"

type Intent int

const (
	IntentUnknown Intent = iota
	IntentMetaBunny
	IntentUserLife
	IntentTaskTech
	IntentExternalFact
)

func DetectIntent(s string) Intent {
	t := normalizeText(s)
	switch {
	// META / self-state (robust to missing spaces)
	case hasAny(t, "wiegeht", "wie geht", "wasdenkst", "was denkst", "hastduangst", "hast du angst", "energie", "cooldown"):
		return IntentMetaBunny

	// External factual questions: pattern-based first (language-agnostic-ish)
	case isExternalFactPattern(t):
		return IntentExternalFact

	case hasAny(t,
		"wetter", "vorhersage", "temperatur", "regen", "wind",
		"news", "nachrichten", "heute", "morgen",
		"kurs", "preis", "aktie", "bitcoin", "ethereum",
		"datum", "uhrzeit", "wann", "wie viel",
		"wer ist", "aktuell", "jetzt",
	):
		return IntentExternalFact
	case hasAny(t, "glücklich", "sinn", "beziehung", "stress", "motivation"):
		return IntentUserLife
	case hasAny(t, "code", "patch", "github", "fski", "go ", "ollama", "sqlite", "module"):
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
	case IntentUserLife:
		return "USER_LIFE"
	case IntentTaskTech:
		return "TASK_TECH"
	default:
		return "GENERAL"
	}
}

func hasAny(t string, subs ...string) bool {
	for _, s := range subs {
		if strings.Contains(t, s) {
			return true
		}
	}
	return false
}

func normalizeText(s string) string {
	t := strings.ToLower(strings.TrimSpace(s))
	// remove spaces for "wiegeht" style matching (keep original too via hasAny list)
	t = strings.ReplaceAll(t, "\t", " ")
	for strings.Contains(t, "  ") {
		t = strings.ReplaceAll(t, "  ", " ")
	}
	return t
}

func isExternalFactPattern(t string) bool {
	// These are generic *question* patterns that imply external factual lookup.
	// Examples:
	// - wo liegt buenos aires
	// - wo ist x
	// - wer ist der ceo von x
	// - was ist die hauptstadt von x
	// - wie hoch ist der preis von x
	// We keep it intentionally broad.
	if strings.Contains(t, "wo liegt") || strings.Contains(t, "wo ist") {
		return true
	}
	if strings.Contains(t, "hauptstadt") {
		return true
	}
	if strings.Contains(t, "wer ist") || strings.Contains(t, "wer war") {
		return true
	}
	if strings.Contains(t, "wie hoch") || strings.Contains(t, "wie viel") {
		return true
	}
	// Simple hint: question mark plus noun-ish query
	if strings.HasSuffix(t, "?") && (strings.Contains(t, "wo ") || strings.Contains(t, "wer ") || strings.Contains(t, "wann ")) {
		return true
	}
	return false
}
