package brain

import (
	"strings"

	"frankenstein-v0/internal/epi"
)

// ApplyUtteranceFilter removes configured phrases from output.
// It is epigenetic (config-driven) and intentionally dumb: no rewriting.
func ApplyUtteranceFilter(out string, eg *epi.Epigenome) string {
	if eg == nil {
		return out
	}
	phrases := eg.UtteranceBannedPhrases()
	if len(phrases) == 0 {
		return out
	}
	t := out
	for _, p := range phrases {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		t = strings.ReplaceAll(t, p, "")
	}
	// cleanup spaces introduced by removals
	for strings.Contains(t, "  ") {
		t = strings.ReplaceAll(t, "  ", " ")
	}
	return strings.TrimSpace(t)
}
