package brain

import (
	"regexp"
	"strconv"
	"strings"

	"frankenstein-v0/internal/epi"
)

// OfflineReflexReply provides a data-driven (epigenetic) fallback response when the LLM backend is offline.
// This avoids scattered hard-coded keyword hacks in the executive.
func OfflineReflexReply(eg *epi.Epigenome, ws *Workspace, userText string) string {
	if eg == nil {
		return "LLM backend offline."
	}
	t := strings.ToLower(strings.TrimSpace(userText))
	rules, def := eg.OfflineReflexRules()
	for _, r := range rules {
		if matchOfflineRule(t, r) {
			return expandOfflineTemplate(r.Reply, ws)
		}
	}
	if strings.TrimSpace(def) != "" {
		return expandOfflineTemplate(def, ws)
	}
	return "LLM backend offline."
}

func matchOfflineRule(t string, r epi.OfflineReflexRule) bool {
	for _, c := range r.Contains {
		c = strings.ToLower(strings.TrimSpace(c))
		if c == "" {
			continue
		}
		if strings.Contains(t, c) {
			return true
		}
	}
	for _, pat := range r.Regex {
		re, err := regexp.Compile(pat)
		if err != nil {
			continue
		}
		if re.MatchString(t) {
			return true
		}
	}
	return false
}

func expandOfflineTemplate(s string, ws *Workspace) string {
	s = strings.TrimSpace(s)
	if ws == nil {
		return s
	}
	energy := ws.EnergyHint
	if energy <= 0 {
		energy = (1.0 - clamp01(ws.DrivesEnergyDeficit)) * 100.0
	}
	s = strings.ReplaceAll(s, "{{energy}}", ftoa1(energy))
	s = strings.ReplaceAll(s, "{{survival}}", ftoa2(clamp01(ws.DrivesEnergyDeficit)))
	s = strings.ReplaceAll(s, "{{resource}}", strings.TrimSpace(ws.ResourceHint))
	return strings.TrimSpace(s)
}

func ftoa1(f float64) string { return strings.TrimSpace(strconv.FormatFloat(f, 'f', 1, 64)) }
func ftoa2(f float64) string { return strings.TrimSpace(strconv.FormatFloat(f, 'f', 2, 64)) }
