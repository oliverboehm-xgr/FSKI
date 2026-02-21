package brain

import (
	"regexp"
	"strings"
)

// Very lightweight candidate extraction.
// Goal: when user asks about a term we don't know -> trigger acquisition.
//
// Examples:
// - "hast du angst?" -> candidate "angst", hint "affect"
// - "hast du scham"  -> "scham", "affect"
// - "was ist epigenetik?" -> "epigenetik", "concept"
var (
	reAffect = regexp.MustCompile(`(?i)\b(hast du|hastdu)\s+([a-zäöüß\-]{3,32})\b`)
	reWasIst = regexp.MustCompile(`(?i)\b(was ist|wasist|erklär|erklaer)\s+(?:mir\s+)?([a-zäöüß\-]{3,48})\b`)
)

func ExtractCandidate(userText string) (term string, hint string) {
	t := strings.TrimSpace(userText)
	if t == "" {
		return "", ""
	}
	if m := reAffect.FindStringSubmatch(t); len(m) >= 3 {
		term = strings.ToLower(m[2])
		return term, "affect"
	}
	if m := reWasIst.FindStringSubmatch(t); len(m) >= 3 {
		term = strings.ToLower(m[2])
		return term, "concept"
	}
	return "", ""
}
