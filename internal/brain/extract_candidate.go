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
// - "wo liegt buenos aires" -> "buenos aires", "location"
// - "wer ist X" -> "X", "entity"
var (
	reAffect   = regexp.MustCompile(`(?i)\b(hast du|hastdu)\s+([a-zäöüß\-]{3,32})\b`)
	reWasIst   = regexp.MustCompile(`(?i)\b(was ist|wasist|erklär|erklaer)\s+(?:mir\s+)?([a-zäöüß\-]{3,48})\b`)
	reWoLiegt  = regexp.MustCompile(`(?i)\b(wo liegt|wo ist)\s+([a-zäöüß\-]{3,64}(?:\s+[a-zäöüß\-]{2,64}){0,3})\b`)
	reWerIst   = regexp.MustCompile(`(?i)\b(wer ist|wer war)\s+([a-zäöüß\-]{3,64}(?:\s+[a-zäöüß\-]{2,64}){0,3})\b`)
	reBedeutet = regexp.MustCompile(`(?i)\b(bedeutet|definition|was bedeutet)\s+([a-zäöüß\-]{3,64})\b`)
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

	if m := reWoLiegt.FindStringSubmatch(t); len(m) >= 3 {
		term = strings.ToLower(strings.TrimSpace(m[2]))
		return term, "location"
	}
	if m := reWerIst.FindStringSubmatch(t); len(m) >= 3 {
		term = strings.ToLower(strings.TrimSpace(m[2]))
		return term, "entity"
	}
	if m := reBedeutet.FindStringSubmatch(t); len(m) >= 3 {
		term = strings.ToLower(strings.TrimSpace(m[2]))
		return term, "concept"
	}
	return "", ""
}
