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
	t := strings.ToLower(s)
	switch {
	case hasAny(t, "wie geht", "was denkst", "hast du angst", "hast du gefühl", "energie", "cooldown"):
		return IntentMetaBunny
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
