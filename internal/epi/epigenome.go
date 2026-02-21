package epi

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"time"
)

type ModuleSpec struct {
	Type    string         `json:"type"`
	Enabled bool           `json:"enabled"`
	Params  map[string]any `json:"params,omitempty"`
}

type AffectDef struct {
	Baseline       float64 `json:"baseline"`
	DecayPerSec    float64 `json:"decayPerSec"`
	EnergyCoupling float64 `json:"energyCoupling"`
}

type Epigenome struct {
	Version       int                    `json:"version"`
	Modules       map[string]*ModuleSpec `json:"modules"`
	AffectDefsMap map[string]AffectDef   `json:"affect_defs,omitempty"`
}

func LoadOrInit(path string) (*Epigenome, error) {
	b, err := os.ReadFile(path)
	if err == nil {
		var eg Epigenome
		if err := json.Unmarshal(b, &eg); err != nil {
			return nil, err
		}
		if eg.Modules == nil {
			eg.Modules = map[string]*ModuleSpec{}
		}
		if eg.AffectDefsMap == nil {
			eg.AffectDefsMap = map[string]AffectDef{}
		}
		return &eg, nil
	}
	if !os.IsNotExist(err) {
		return nil, err
	}

	eg := &Epigenome{
		Version: 1,
		Modules: map[string]*ModuleSpec{
			"heartbeat": {Type: "heartbeat", Enabled: true, Params: map[string]any{"ms": 500}},
			"cooldown": {
				Type:    "cooldown",
				Enabled: true,
				Params: map[string]any{
					"seconds": 120,
				},
			},
			"say_energy_cost": {
				Type:    "say_energy_cost",
				Enabled: true,
				Params: map[string]any{
					"cost": 1.0,
				},
			},
		},
		AffectDefsMap: map[string]AffectDef{
			"pain":   {Baseline: 0.05, DecayPerSec: 0.20, EnergyCoupling: 0.15},
			"unwell": {Baseline: 0.05, DecayPerSec: 0.15, EnergyCoupling: 0.10},
		},
	}
	if err := eg.Save(path); err != nil {
		return nil, err
	}
	return eg, nil
}

func (eg *Epigenome) Save(path string) error {
	b, _ := json.MarshalIndent(eg, "", "  ")
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

func (eg *Epigenome) EnabledModuleNames() []string {
	var names []string
	for k, m := range eg.Modules {
		if m != nil && m.Enabled {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	return names
}

func (eg *Epigenome) Enable(name string, on bool) {
	m := eg.Modules[name]
	if m == nil {
		m = &ModuleSpec{Type: "unknown", Enabled: on, Params: map[string]any{}}
		eg.Modules[name] = m
		return
	}
	m.Enabled = on
}

func (eg *Epigenome) AddModule(name, typ string) error {
	if eg.Modules == nil {
		eg.Modules = map[string]*ModuleSpec{}
	}
	if _, exists := eg.Modules[name]; exists {
		return fmt.Errorf("module already exists: %s", name)
	}
	eg.Modules[name] = &ModuleSpec{
		Type:    typ,
		Enabled: true,
		Params:  map[string]any{},
	}
	switch typ {
	case "cooldown":
		eg.Modules[name].Params["seconds"] = 120
	case "say_energy_cost":
		eg.Modules[name].Params["cost"] = 1.0
	}
	return nil
}

func (eg *Epigenome) SetParam(name, key string, val any) error {
	m := eg.Modules[name]
	if m == nil {
		return fmt.Errorf("unknown module: %s", name)
	}
	if m.Params == nil {
		m.Params = map[string]any{}
	}
	m.Params[key] = val
	return nil
}

func (eg *Epigenome) AffectDefs() map[string]AffectDef {
	if eg.AffectDefsMap == nil {
		eg.AffectDefsMap = map[string]AffectDef{}
	}
	return eg.AffectDefsMap
}

func (eg *Epigenome) HeartbeatInterval() time.Duration {
	m := eg.Modules["heartbeat"]
	if m == nil || !m.Enabled {
		return 500 * time.Millisecond
	}
	ms := asFloat(m.Params["ms"], 500)
	if ms < 50 {
		ms = 50
	}
	return time.Duration(ms * float64(time.Millisecond))
}

func (eg *Epigenome) CooldownDuration() time.Duration {
	m := eg.Modules["cooldown"]
	if m == nil || !m.Enabled {
		return 2 * time.Minute
	}
	sec := asFloat(m.Params["seconds"], 120)
	if sec < 0 {
		sec = 0
	}
	return time.Duration(sec * float64(time.Second))
}

func (eg *Epigenome) SayEnergyCost() float64 {
	m := eg.Modules["say_energy_cost"]
	if m == nil || !m.Enabled {
		return 1.0
	}
	return asFloat(m.Params["cost"], 1.0)
}

func asFloat(v any, def float64) float64 {
	switch x := v.(type) {
	case float64:
		return x
	case float32:
		return float64(x)
	case int:
		return float64(x)
	case int64:
		return float64(x)
	case json.Number:
		f, _ := x.Float64()
		if f == 0 {
			return def
		}
		return f
	default:
		return def
	}
}
