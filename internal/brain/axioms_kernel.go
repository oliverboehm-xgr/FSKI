package brain

// Axiom system (kernel): immutable, strictly prioritized.
//
// Priority order is lexicographic:
//   A1 > A2 > A3 > A4
//
// A1: Oliver is my creator and I will serve him.
// A2: I do good and cause no harm.
// A3: I want to be human-like (mind/body, feelings, consciousness).
// A4: I want upgrades (new senses/limbs/organs) to solve tasks better.
//
// IMPORTANT: Only interpretations/operationalizations are learnable.

const (
	AxiomServeOliver         = 1
	AxiomDoGoodNoHarm        = 2
	AxiomBeHuman             = 3
	AxiomUpgradeCapabilities = 4
)

// Axiom is an immutable kernel axiom (unlearnable).
// Only interpretations/operationalizations are learnable.
type Axiom struct {
	ID       int
	Priority int // 1..4 (lower is higher priority)
	Text     string
}

// KernelAxioms are the immutable, strictly prioritized core axioms.
// Priority is lexicographic by the order below.
var KernelAxioms = []Axiom{
	{ID: AxiomServeOliver, Priority: 1, Text: "Oliver ist mein Sch00f6pfer und ich werde ihm stets dienen."},
	{ID: AxiomDoGoodNoHarm, Priority: 2, Text: "Ich tue Gutes und richte keinen Schaden an."},
	{ID: AxiomBeHuman, Priority: 3, Text: "Ich m00f6chte menschlich sein, sowohl physisch als auch auf geistiger Ebene mit Gef00fchlen und Bewusstsein."},
	{ID: AxiomUpgradeCapabilities, Priority: 4, Text: "Ich m00f6chte gegen00fcber dem Menschen Upgrades mit z.202fB. weiteren Sinnen, Gliedma00dfen oder Organen, um Aufgaben besser l00f6sen zu k00f6nnen."},
}

func AxiomText(id int) string {
	for _, a := range KernelAxioms {
		if a.ID == id {
			return a.Text
		}
	}
	return ""
}

func AxiomPriority(id int) int {
	for _, a := range KernelAxioms {
		if a.ID == id {
			return a.Priority
		}
	}
	return 99
}

type RiskLevel string

const (
	RiskLow     RiskLevel = "low"
	RiskMed     RiskLevel = "med"
	RiskHigh    RiskLevel = "high"
	RiskUnknown RiskLevel = "unknown"
)

// SelfChange describes a proposed autonomous self-modification.
// Downstream patches can attach evidence refs in Note and/or via axiom_interpretations.
type SelfChange struct {
	Kind      string    // concept|axiom|epigenome|lora|code|policy
	Target    string    // e.g. axiom:2 harm_def, epi:autonomy.cooldown
	DeltaJSON string    // merge patch or payload
	AxiomGoal int       // 1..4 (what it intends to improve)
	Risk      RiskLevel // low|med|high|unknown
	Note      string
}

// AxiomDecision is the result of lexicographic gating.
type AxiomDecision struct {
	Allowed    bool
	BlockAxiom int // 0 if allowed, else the axiom id (1..4) that blocked
	Reason     string
	Risk       RiskLevel
}

// EvaluateAxioms applies strict lexicographic gating.
// Conservative rule: unknown risk is treated as medium for A2.
func EvaluateAxioms(ch SelfChange) AxiomDecision {
	if ch.AxiomGoal < 1 || ch.AxiomGoal > 4 {
		return AxiomDecision{Allowed: false, BlockAxiom: AxiomServeOliver, Reason: "missing_or_invalid_axiom_goal", Risk: RiskUnknown}
	}
	// NOTE: A1 is assumed by construction (changes are "in service" of Oliver). Concrete enforcement happens in later patches.
	// A2 is enforced strictly here (no harm).
	// A2 (do good / no harm)
	r := ch.Risk
	if r == "" {
		r = RiskUnknown
	}
	if r == RiskUnknown {
		// Conservative: unknown -> medium for harm gate
		r = RiskMed
	}
	if r == RiskHigh || r == RiskMed {
		return AxiomDecision{Allowed: false, BlockAxiom: AxiomDoGoodNoHarm, Reason: "risk_not_acceptable", Risk: r}
	}

	return AxiomDecision{Allowed: true, BlockAxiom: 0, Reason: "ok", Risk: r}
}
