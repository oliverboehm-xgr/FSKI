package brain

import "strings"

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

func DetectIntent(s string) Intent {
	t := normalizeText(s)

	switch {
	case hasAny(t, "wiegeht", "wie geht", "wasdenkst", "was denkst", "hastduangst", "hast du angst", "energie", "cooldown"):
		return IntentMetaBunny

	// opinion must win over external facts/research patterns
	case isOpinionRequest(t):
		return IntentOpinion

	// explicit research command (meta)
	case isResearchCommand(t):
		return IntentResearchCommand

	// external factual patterns
	case isExternalFactPattern(t):
		return IntentExternalFact

	case hasAny(t,
		"wetter", "vorhersage", "temperatur", "regen", "wind",
		"news", "nachrichten", "heute", "morgen",
		"kurs", "preis", "aktie", "bitcoin", "ethereum",
		"datum", "uhrzeit", "wann", "wie viel",
		"wer ist", "aktuell", "jetzt",
		"krieg", "konflikt", // generisch, kein Topic-hardcode
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
	t = strings.ReplaceAll(t, "\t", " ")
	for strings.Contains(t, "  ") {
		t = strings.ReplaceAll(t, "  ", " ")
	}
	return t
}

func isResearchCommand(t string) bool {
	return hasAny(t,
		"recherchier", "recherchiere",
		"im internet", "nachsehen", "schau nach",
		"google", "web", "quelle", "quellen",
	)
}

func isOpinionRequest(t string) bool {
	return hasAny(t,
		"deine meinung", "meine meinung", "meinung",
		"was sagst du", "äußere dich", "aeussere dich",
		"bewerte", "einschätzung", "einschaetzung",
		"position", "haltung",
		"gut oder schlecht", "findest du es gut", "findest du es schlecht",
		"stell dich", "stellung beziehen", "partei ergreifen",
		"reagiere so wie du willst", "reagiere wie du willst",
	)
}

func isExternalFactPattern(t string) bool {
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
	if strings.HasSuffix(t, "?") && (strings.Contains(t, "wo ") || strings.Contains(t, "wer ") || strings.Contains(t, "wann ")) {
		return true
	}
	return false
}
