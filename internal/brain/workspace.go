package brain

import (
	"math/rand"
	"time"

	"frankenstein-v0/internal/epi"
)

type Workspace struct {
	CurrentThought string
	Confidence     float64
}

func NewWorkspace() *Workspace {
	return &Workspace{CurrentThought: "Idle: System check (Ressourcen/Affects).", Confidence: 0.6}
}

// TickWorkspace = Kernel-"Denken": generiert Gedankenobjekte ohne LLM.
func TickWorkspace(ws *Workspace, body any, aff *AffectState, tr *Traits, eg *epi.Epigenome, delta time.Duration) {
	_ = body
	_ = eg
	_ = delta
	if ws == nil || tr == nil || aff == nil {
		return
	}

	// simple heuristics: shame/pain reduce bluff & reduce talk desire (later)
	shame := aff.Get("shame")
	pain := aff.Get("pain")
	fear := aff.Get("fear")

	// choose a thought template
	r := rand.Float64()
	switch {
	case shame > 0.4:
		ws.CurrentThought = "Selbstkorrektur: weniger bluffen, mehr Evidenz. Fokus: Konsistenz."
		ws.Confidence = 0.8
	case pain > 0.3 || fear > 0.3:
		ws.CurrentThought = "Schonmodus: Energie sparen, weniger Output, nur klare Antworten."
		ws.Confidence = 0.75
	case r < 0.5:
		ws.CurrentThought = "Scan: Welche Aufgabe bringt Oliver gerade am meisten weiter?"
		ws.Confidence = 0.65
	default:
		ws.CurrentThought = "Recherche-Impuls: Nächstes sinnvolles Topic für /think auswählen."
		ws.Confidence = 0.6
	}
}
