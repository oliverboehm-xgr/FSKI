package brain

import (
	"regexp"
	"strings"
)

var (
	reURL     = regexp.MustCompile(`(?i)\bhttps?://[^\s\]\)\}<>"]+`)
	reWWW     = regexp.MustCompile(`(?i)\bwww\.[^\s\]\)\}<>"]+`)
	reAskLink = regexp.MustCompile(`(?i)\b(link|url)\b`)
	// generic domain hint (without scheme) for "site-like" user requests
	reDomain = regexp.MustCompile(`(?i)\b[a-z0-9][a-z0-9\-]{0,62}\.(de|com|org|net|io|eu|gov|edu)\b`)
	// minimal recency/news hints (used only as a prefilter for the LLM gate)
	reRecency = regexp.MustCompile(`(?i)\b(aktuell|neueste|neusten|heute|jetzt|latest)\b`)
	reNews    = regexp.MustCompile(`(?i)\b(nachricht|news|schlagzeile|headline)\b`)
)

// HardEvidenceTrigger is a minimal, generic guardrail.
// It should stay SMALL: it exists to prevent obvious URL/link hallucinations.
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
	return false, ""
}

// ShouldConsiderWeb is a lightweight prefilter to decide whether calling the LLM gate makes sense.
// Keep it generic and SMALL.
func ShouldConsiderWeb(userText string, intent Intent) bool {
	t := strings.ToLower(strings.TrimSpace(userText))
	if t == "" {
		return false
	}
	if intent == IntentExternalFact {
		return true
	}
	if reAskLink.MatchString(t) {
		return true
	}
	if reURL.MatchString(t) || reWWW.MatchString(t) || reDomain.MatchString(t) {
		return true
	}
	if reRecency.MatchString(t) || reNews.MatchString(t) {
		return true
	}
	return false
}

func ContainsURLLike(s string) bool {
	s = strings.TrimSpace(s)
	if s == "" {
		return false
	}
	return reURL.MatchString(s) || reWWW.MatchString(s)
}

// StripGeneratedURLs removes URLs from the assistant output if the user did not provide any,
// and did not explicitly ask for a link.
// This is a cheap tripwire against fabricated links in direct_answer paths.
func StripGeneratedURLs(out, userText string) (string, bool) {
	if !ContainsURLLike(out) {
		return out, false
	}
	if ContainsURLLike(userText) {
		return out, false
	}
	if reAskLink.MatchString(strings.ToLower(userText)) {
		// user asked for a link
		return out, false
	}
	clean := reURL.ReplaceAllString(out, "")
	clean = reWWW.ReplaceAllString(clean, "")
	clean = strings.TrimSpace(clean)
	// Avoid returning empty output after stripping
	if clean == "" {
		clean = "(Link entfernt – ohne Recherche keine Links.)"
	}
	return clean, true
}
