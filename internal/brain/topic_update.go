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
	if isLikelyContextFollowup(t) {
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

	// Only shift implicitly on clearly long turns to avoid drift on short follow-ups.
	if len([]rune(t)) >= 48 && cand != ws.ActiveTopic {
		ws.ActiveTopic = cand
	}
	return ws.ActiveTopic
}

func isLikelyContextFollowup(t string) bool {
	if t == "" {
		return false
	}
	r := []rune(t)
	if len(r) > 80 {
		return false
	}
	cues := []string{"dazu", "darüber", "darueber", "davon", "vorhin", "oben", "nochmal", "genannte", "letzte", "dies", "diese", "diesen", "dem", "den"}
	for _, c := range cues {
		if strings.Contains(t, c) {
			return true
		}
	}
	return false
}

func isMetaTopic(t string) bool {
	switch strings.ToLower(t) {
	case "jetzt", "moment", "gegenwart", "heute", "morgen":
		return true
	default:
		return false
	}
}
