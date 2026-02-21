package brain

import "strings"

// PostprocessGerman removes overly eager boilerplate and punctuation noise.
func PostprocessGerman(s string) string {
	t := strings.TrimSpace(s)
	repl := []struct{ a, b string }{
		{"Ich bin online und bereit", "Ich bin da"},
		{"Wie kann ich dir helfen?", "Worum geht’s?"},
		{"Wie kann ich Ihnen helfen?", "Worum geht’s?"},
		{"Ich bin bereit,", ""},
		{"!", ""},
	}
	for _, r := range repl {
		t = strings.ReplaceAll(t, r.a, r.b)
	}
	t = strings.ReplaceAll(t, "\r\n", "\n")
	for strings.Contains(t, "\n\n\n") {
		t = strings.ReplaceAll(t, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(t)
}
