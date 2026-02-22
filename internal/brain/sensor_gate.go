package brain

import (
	"regexp"
	"strings"
)

var (
	reURL     = regexp.MustCompile(`(?i)\bhttps?://[^\s\]\)\}<>\"]+`)
	reWWW     = regexp.MustCompile(`(?i)\bwww\.[^\s\]\)\}<>\"]+`)
	reDomain  = regexp.MustCompile(`(?i)\b[a-z0-9][a-z0-9\-]{0,62}\.(de|com|org|net|io|eu|gov|edu)\b`)
	reAskLink = regexp.MustCompile(`(?i)\b(link|url)\b`)
	// very small recency / news hints as fallback when the LLM gate is unavailable
	reRecency = regexp.MustCompile(`(?i)\b(aktuell|aktuellste|neueste|heute|jetzt|latest)\b`)
	reNews    = regexp.MustCompile(`(?i)\b(nachricht|news|schlagzeile|headline|zeitstempel)\b`)
)

// HardEvidenceTrigger is a minimal, generic guardrail.
// It must stay SMALL: it exists to prevent obvious link/time-stamp hallucinations.
func HardEvidenceTrigger(userText string) (bool, string) {
	t := strings.ToLower(strings.TrimSpace(userText))
	if t == "" {
		return false, ""
	}
	if reAskLink.MatchString(t) {
		return true, "ask_link"
	}
	if reURL.MatchString(t) || reWWW.MatchString(t) {
		return true, "contains_url"
	}
	if reDomain.MatchString(t) {
		return true, "mentions_domain"
	}
	// fallback: explicit recency/news asks
	if reRecency.MatchString(t) && reNews.MatchString(t) {
		return true, "recency_news"
	}
	return false, ""
}

func containsURLLike(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	return reURL.MatchString(s) || reWWW.MatchString(s)
}

// StripGeneratedURLs removes URLs from assistant output if the user did not provide any
// and did not explicitly ask for a link. Cheap tripwire against fabricated links.
func StripGeneratedURLs(out, userText string) (string, bool) {
	if !containsURLLike(out) {
		return out, false
	}
	if containsURLLike(userText) {
		return out, false
	}
	if reAskLink.MatchString(strings.ToLower(userText)) {
		return out, false
	}
	clean := reURL.ReplaceAllString(out, "")
	clean = reWWW.ReplaceAllString(clean, "")
	clean = strings.TrimSpace(clean)
	if clean == "" {
		clean = "(Link entfernt – ohne Recherche keine Links.)"
	}
	return clean, true
}
