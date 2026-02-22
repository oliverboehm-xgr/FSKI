package brain

import (
	"strings"
	"unicode"
)

// TokenizeAlphaNumLower is a generic tokenizer for topic anchoring and simple heuristics.
func TokenizeAlphaNumLower(s string) []string {
	s = strings.TrimSpace(strings.ToLower(s))
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		out = append(out, b.String())
		b.Reset()
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}
