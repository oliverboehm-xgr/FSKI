package brain

import (
	"math/rand"
	"time"

	"frankenstein-v0/internal/epi"
)

type Workspace struct {
	CurrentThought string
	Confidence     float64
	LastTopic      string
	PrevUserText   string
	LastUserText   string
	ActiveTopic    string
	_daydreamAccum float64
	LastLatencyMs  float64
	LatencyEMA     float64
	lastTuneAt     time.Time

	// Human-like thinking: images + inner speech
	VisualScene    string
	InnerSpeech    string
	LastDaydreamAt time.Time

	// Hint from body for areas that can't access BodyState directly
	EnergyHint float64

	// Drives v1 integration hints (set by main each tick)
	DrivesEnergyDeficit float64 // survival
	SocialCraving       float64 // 1-soc_sat
	UrgeInteractHint    float64 // from DrivesV1.UrgeInteract
	ResourceHint        string  // human-readable measured resource numbers
	LastHelpAt          time.Time

	// Social ping throttle
	LastSocialPingAt time.Time

	// Router bookkeeping for learning
	LastRoutedIntent string

	// UI plumbing: last persisted user message id (so training trials can link it)
	LastUserMsgID int64

	// Training: when true, candidate generation should avoid side-effects (no concept acquisition, no web, etc.)
	TrainingDryRun bool

	// Policy selection (bandit)
	LastPolicyCtx    string
	LastPolicyAction string
	LastPolicyStyle  string

	// Executive / Survival-gate controls (kernel truth)
	SurvivalMode    bool
	WebAllowed      bool
	AutonomyAllowed bool
	MaxContextTurns int
	MaxDetailItems  int

	// Executive plan (minimal FSM)
	ActiveGoal      string
	ActivePlanTopic string
	PlanSteps       []string
	PlanIndex       int

	// Generic info-gate traces (debug/telemetry)
	LastUserInfoScore float64
	LastUserTopToken  string
	LLMAvailable      bool
	// Non-fatal: configured Ollama models that are missing (used for graceful routing)
	OLLAMAMissing []string

	// Cortex sensor-gate traces (per turn)
	LastSenseNeedWeb bool
	LastSenseScore   float64
	LastSenseQuery   string
	LastSenseReason  string
	LastSenseText    string
	// Teleology: short axiom operationalization snippet (rules/metrics/defs) 
	AxiomContext string 
}

func NewWorkspace() *Workspace {
	return &Workspace{CurrentThought: "Idle: Systemcheck (Ressourcen/Affects).", Confidence: 0.6, LastTopic: "", ActiveTopic: ""}
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

var topicStopwords = map[string]struct{}{"bist": {}, "bin": {}, "sind": {}, "seid": {}, "sein": {}, "hast": {}, "habt": {}, "hat": {}, "haben": {}, "ja": {}, "ok": {}, "okay": {}, "bitte": {}, "danke": {}, "nochmal": {}, "wieder": {}, "genau": {}, "klar": {}, "mach": {}, "machst": {}, "machstdu": {}, "du": {}, "ich": {}, "wir": {}, "ihr": {}, "sie": {}, "er": {}, "es": {}}

func ExtractTopic(s string) string {
	toks := TokenizeAlphaNumLower(s)
	best := ""
	for _, t := range toks {
		if len(t) < 3 {
			continue
		}
		if _, bad := topicStopwords[t]; bad {
			continue
		}
		if len(t) >= len(best) {
			best = t
		}
	}
	return best
}
