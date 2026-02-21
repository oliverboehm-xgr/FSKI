package main

import (
	"fmt"
	"strings"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
)

func buildSelfLines(sm *epi.SelfModel, aff *brain.AffectState) string {
	var b strings.Builder
	b.WriteString("NOTE: Use ONLY these numbers if you mention them.\n")
	if sm != nil {
		b.WriteString(fmt.Sprintf("BODY: energy=%.1f/%0.1f web_count_hour=%d cooldown_until=%s\n",
			sm.Body.Energy,
			sm.Body.EnergyMax,
			sm.Body.WebCountHour,
			sm.Body.Cooldown,
		))
		b.WriteString(fmt.Sprintf("WORKSPACE: confidence=%.3f current_thought=%q\n", sm.Workspace.Confidence, sm.Workspace.CurrentThought))
		b.WriteString(fmt.Sprintf("TRAITS: bluff_rate=%.3f honesty_bias=%.3f\n", sm.Traits.BluffRate, sm.Traits.HonestyBias))
	}
	if aff != nil {
		b.WriteString("AFFECTS:")
		for _, k := range aff.Keys() {
			b.WriteString(fmt.Sprintf(" %s=%.3f", k, aff.Get(k)))
		}
		b.WriteString("\n")
	}
	return b.String()
}
