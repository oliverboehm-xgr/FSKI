package brain

import (
	"sort"
	"time"

	"frankenstein-v0/internal/epi"
)

// AffectState is generic: bunny can add new affects at runtime via epigenome config.
// Values are 0..1 floats (you can exceed later, but keep it bounded for now).
type AffectState struct {
	m map[string]float64
}

func NewAffectState() *AffectState {
	return &AffectState{m: map[string]float64{}}
}

func (a *AffectState) Ensure(key string, init float64) {
	if _, ok := a.m[key]; !ok {
		a.m[key] = clamp01(init)
	}
}

func (a *AffectState) Get(key string) float64    { return a.m[key] }
func (a *AffectState) Set(key string, v float64) { a.m[key] = clamp01(v) }

func (a *AffectState) Keys() []string {
	keys := make([]string, 0, len(a.m))
	for k := range a.m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// TickAffects: homeostasis loop. No LLM involved.
// BodyState is passed as interface to avoid import cycles; we only use energy heuristics via epi helpers (next patch can formalize shared types).
func TickAffects(body any, a *AffectState, eg *epi.Epigenome, delta time.Duration) {
	defs := eg.AffectDefs()
	dt := delta.Seconds()
	if dt <= 0 {
		return
	}

	for name, d := range defs {
		a.Ensure(name, d.Baseline)
	}

	energy := epi.ExtractEnergy(body)
	energy01 := clamp01(energy / 100.0)

	for name, d := range defs {
		v := a.Get(name)
		v += (d.Baseline - v) * clamp01(d.DecayPerSec*dt)

		if d.EnergyCoupling != 0 {
			v += (1.0 - energy01) * d.EnergyCoupling * dt
		}
		a.Set(name, v)
	}
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}
