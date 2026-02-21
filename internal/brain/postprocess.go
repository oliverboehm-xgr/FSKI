package brain

import "strings"

// PostprocessGerman: conservative cleanup only.
// IMPORTANT: must NOT delete meaning or sentence starts.
func PostprocessGerman(s string) string {
	t := strings.TrimSpace(s)
	// Very targeted replacements (full phrases only)
	t = strings.ReplaceAll(t, "Ich bin online und bereit, Fragen zu beantworten.", "Ich bin da. Worum geht’s?")
	t = strings.ReplaceAll(t, "Ich bin online und bereit.", "Ich bin da.")
	t = strings.ReplaceAll(t, "Wie kann ich dir helfen?", "Worum geht’s?")
	t = strings.ReplaceAll(t, "Wie kann ich Ihnen helfen?", "Worum geht’s?")

	// Do NOT strip "!" globally. Only trim trailing "!" if it's excessive.
	for strings.HasSuffix(t, "!!") {
		t = strings.TrimSuffix(t, "!")
	}

	t = strings.ReplaceAll(t, "\r\n", "\n")
	for strings.Contains(t, "\n\n\n") {
		t = strings.ReplaceAll(t, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(t)
}
