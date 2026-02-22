package brain

import "strings"

// UpdateActiveTopic is intentionally conservative.
// It updates the active topic only when:
// - active topic is empty, or
// - user explicitly signals topic ("es geht um", "thema", "topic"), or
// - user message is long and contains a strong extracted token.
func UpdateActiveTopic(ws *Workspace, userText string) string {
	if ws == nil {
		return ""
	}
	t := strings.ToLower(strings.TrimSpace(userText))
	if t == "" {
		return ws.ActiveTopic
	}
	if isFollowupReference(t) {
		return ws.ActiveTopic
	}

	cand := ExtractTopic(userText)
	if cand == "" || isMetaTopic(cand) {
		return ws.ActiveTopic
	}

	explicit := strings.Contains(t, "es geht um") ||
		strings.Contains(t, "thema") ||
		strings.Contains(t, "topic") ||
		strings.Contains(t, "wir reden über") ||
		strings.Contains(t, "wir reden ueber")

	if ws.ActiveTopic == "" {
		ws.ActiveTopic = cand
		return ws.ActiveTopic
	}

	if explicit {
		ws.ActiveTopic = cand
		return ws.ActiveTopic
	}

	// If message is long-ish and candidate differs, allow shift.
	if len([]rune(t)) >= 28 && cand != ws.ActiveTopic {
		ws.ActiveTopic = cand
	}
	return ws.ActiveTopic
}

func isFollowupReference(t string) bool {
	if t == "" {
		return false
	}
	if !(strings.Contains(t, "nachricht") || strings.Contains(t, "punkt") || strings.Contains(t, "nummer") || strings.Contains(t, "nr.")) {
		return false
	}
	hasDigit := false
	for _, r := range t {
		if r >= '0' && r <= '9' {
			hasDigit = true
			break
		}
	}
	if !hasDigit {
		return false
	}
	return strings.Contains(t, "lass uns") || strings.Contains(t, "sprechen") || strings.Contains(t, "weiter") || strings.Contains(t, "darüber") || strings.Contains(t, "darueber")
}

func isMetaTopic(t string) bool {
	switch strings.ToLower(t) {
	case "jetzt", "moment", "gegenwart", "heute", "morgen":
		return true
	default:
		return false
	}
}
