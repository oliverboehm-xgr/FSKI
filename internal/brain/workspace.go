package brain

import (
	"math/rand"
	"regexp"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
)

type Workspace struct {
	CurrentThought string
	Confidence     float64
	LastTopic      string
	LastUserText   string
	PrevUserText   string
	_daydreamAccum float64
}

func NewWorkspace() *Workspace {
	return &Workspace{CurrentThought: "Idle: Systemcheck (Ressourcen/Affects).", Confidence: 0.6, LastTopic: ""}
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

	if ws.LastTopic != "" {
		if shame > 0.3 {
			ws.CurrentThought = "Fokus: sauber bleiben. Erst klären, dann antworten. Thema: " + ws.LastTopic
			ws.Confidence = 0.8
			return
		}
		if pain > 0.25 || fear > 0.25 {
			ws.CurrentThought = "Schonmodus: kurz, präzise. Thema: " + ws.LastTopic
			ws.Confidence = 0.75
			return
		}
		ws.CurrentThought = "Nächster sinnvoller Schritt: 1 Rückfrage stellen. Thema: " + ws.LastTopic
		ws.Confidence = 0.7
		return
	}

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
		ws.CurrentThought = "Scan: Was ist Olivers konkrete Frage, und was fehlt an Kontext?"
		ws.Confidence = 0.65
	default:
		ws.CurrentThought = "Recherche-Impuls: Nächstes sinnvolles Topic für /think auswählen (nur wenn gefragt)."
		ws.Confidence = 0.6
	}
}

var topicRe = regexp.MustCompile(`(?i)\b(wetter|glücklich|stress|beziehung|code|patch|github|fski|ollama|sqlite|xgr|xda?la|mutat|modul|schlaf|angst)\b`)

func ExtractTopic(s string) string {
	m := topicRe.FindStringSubmatch(s)
	if len(m) == 0 {
		return ""
	}
	return strings.ToLower(m[1])
}
