package brain

import "strings"

// PostprocessUtterance: STRUCTURE ONLY.
// No phrase replacements, no semantic edits.
func PostprocessUtterance(s string) string {
	t := strings.TrimSpace(s)
	t = strings.ReplaceAll(t, "\r\n", "\n")
	for strings.Contains(t, "\n\n\n") {
		t = strings.ReplaceAll(t, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(t)
}
