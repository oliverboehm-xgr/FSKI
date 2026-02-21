package epi

import (
	"encoding/json"
	"time"
)

type SelfModel struct {
	Body struct {
		Energy       float64 `json:"energy"`
		EnergyMax    float64 `json:"energyMax"`
		EnergyUnit   string  `json:"energyUnit"`
		WebCountHour int     `json:"webCountHour"`
		Cooldown     string  `json:"cooldownUntil"`
	} `json:"body"`
	Affects   map[string]float64 `json:"affects,omitempty"`
	Epigenome struct {
		EnabledModules []string `json:"enabledModules"`
		Version        int      `json:"version"`
		AffectDefs     any      `json:"affectDefs,omitempty"`
		Lang           string   `json:"lang"`
	} `json:"epigenome"`
	Workspace struct {
		CurrentThought string  `json:"current_thought"`
		Confidence     float64 `json:"confidence"`
	} `json:"workspace"`
	Traits struct {
		BluffRate   float64 `json:"bluff_rate"`
		HonestyBias float64 `json:"honesty_bias"`
	} `json:"traits"`
}

// BuildSelfModel is intentionally minimal for v0.1.
// Later: add runtime stats (mem/cpu/disk), message queue, interests, etc.
type AffectReader interface {
	Keys() []string
	Get(string) float64
}

func BuildSelfModel(body any, aff AffectReader, ws any, tr any, eg *Epigenome) *SelfModel {
	sm := &SelfModel{}
	sm.Epigenome.EnabledModules = eg.EnabledModuleNames()
	sm.Epigenome.Version = eg.Version
	sm.Epigenome.Lang = eg.Lang()

	sm.Body.Energy = ExtractEnergy(body)
	sm.Body.EnergyMax = eg.EnergyMax()
	sm.Body.EnergyUnit = "Energiepunkte (0..energyMax)"
	sm.Body.WebCountHour = ExtractWebCountHour(body)
	sm.Body.Cooldown = ExtractCooldown(body).Format(time.RFC3339)

	if aff != nil {
		sm.Affects = map[string]float64{}
		for _, k := range aff.Keys() {
			sm.Affects[k] = aff.Get(k)
		}
	}
	if ws != nil {
		raw, _ := json.Marshal(ws)
		var tmp struct {
			CurrentThought string  `json:"CurrentThought"`
			Confidence     float64 `json:"Confidence"`
		}
		_ = json.Unmarshal(raw, &tmp)
		sm.Workspace.CurrentThought = tmp.CurrentThought
		sm.Workspace.Confidence = tmp.Confidence
	}
	if tr != nil {
		raw, _ := json.Marshal(tr)
		var tmp struct {
			BluffRate   float64 `json:"BluffRate"`
			HonestyBias float64 `json:"HonestyBias"`
		}
		_ = json.Unmarshal(raw, &tmp)
		sm.Traits.BluffRate = tmp.BluffRate
		sm.Traits.HonestyBias = tmp.HonestyBias
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
