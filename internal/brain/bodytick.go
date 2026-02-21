package brain

import (
	"time"

	"frankenstein-v0/internal/epi"
)

// TickBody is intentionally conservative: it only handles small recovery drift.
// More complex physiology comes later (sleep debt, electrolyte, etc.).
func TickBody(body any, eg *epi.Epigenome, delta time.Duration) {
	_ = eg
	energy := epi.ExtractEnergy(body)
	energy += (delta.Seconds() * 0.02) // +0.02 per sec => +1.2/min
	if energy > 100 {
		energy = 100
	}
	epi.InjectEnergy(body, energy)
	_ = time.Now()
}
