package brain

import "strings"

// NormalizeSearchQuery keeps it simple:
// - trims
// - removes command-like prefixes
// - avoids ultra-short queries
func NormalizeSearchQuery(userText string) string {
	q := strings.TrimSpace(userText)
	q = strings.TrimPrefix(q, "/say")
	q = strings.TrimSpace(q)
	// very small normalization
	q = strings.ReplaceAll(q, "  ", " ")
	// If user asks "wo liegt X" -> use that directly (good search query)
	// If user asks "wie wird das wetter morgen in deutschland" -> also fine as-is.
	if len([]rune(q)) < 8 {
		// too short: keep as-is; caller will likely ask clarification if search fails
		return q
	}
	return q
}
