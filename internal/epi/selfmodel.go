package epi

import (
	"encoding/json"
	"time"
)

type SelfModel struct {
	Body struct {
		Energy       float64 `json:"energy"`
		WebCountHour int     `json:"webCountHour"`
		Cooldown     string  `json:"cooldownUntil"`
	} `json:"body"`
	Affects   map[string]float64 `json:"affects,omitempty"`
	Epigenome struct {
		EnabledModules []string `json:"enabledModules"`
		Version        int      `json:"version"`
		AffectDefs     any      `json:"affectDefs,omitempty"`
	} `json:"epigenome"`
}

// BuildSelfModel is intentionally minimal for v0.1.
// Later: add runtime stats (mem/cpu/disk), message queue, interests, etc.
type AffectReader interface {
	Keys() []string
	Get(string) float64
}

func BuildSelfModel(body any, aff AffectReader, eg *Epigenome) *SelfModel {
	sm := &SelfModel{}
	sm.Epigenome.EnabledModules = eg.EnabledModuleNames()
	sm.Epigenome.Version = eg.Version

	sm.Body.Energy = ExtractEnergy(body)
	sm.Body.WebCountHour = ExtractWebCountHour(body)
	sm.Body.Cooldown = ExtractCooldown(body).Format(time.RFC3339)

	if aff != nil {
		sm.Affects = map[string]float64{}
		for _, k := range aff.Keys() {
			sm.Affects[k] = aff.Get(k)
		}
	}
	sm.Epigenome.AffectDefs = eg.AffectDefs()
	return sm
}

func ExtractEnergy(body any) float64 {
	switch b := body.(type) {
	case interface{ GetEnergy() float64 }:
		return b.GetEnergy()
	default:
		raw, _ := json.Marshal(body)
		var tmp struct {
			Energy float64 `json:"Energy"`
		}
		_ = json.Unmarshal(raw, &tmp)
		if tmp.Energy == 0 {
			return 50
		}
		return tmp.Energy
	}
}

func InjectEnergy(body any, v float64) {
	type setEnergy interface{ SetEnergy(float64) }
	if b, ok := body.(setEnergy); ok {
		b.SetEnergy(v)
	}
}

func ExtractWebCountHour(body any) int {
	raw, _ := json.Marshal(body)
	var tmp struct {
		WebCountHour int `json:"WebCountHour"`
	}
	_ = json.Unmarshal(raw, &tmp)
	return tmp.WebCountHour
}

func ExtractCooldown(body any) time.Time {
	raw, _ := json.Marshal(body)
	var tmp struct {
		CooldownUntil time.Time `json:"CooldownUntil"`
	}
	_ = json.Unmarshal(raw, &tmp)
	return tmp.CooldownUntil
}
