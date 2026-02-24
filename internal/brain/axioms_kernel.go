package brain

// Axiom system (kernel): immutable, strictly prioritized.
// Each axiom is an independent goal; ONLY higher-priority axioms act as constraints.
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

// EvaluateAxioms applies lexicographic constraint gating:
// - Each self-change has a single AxiomGoal (1..4) it intends to improve.
// - We ONLY check whether the change violates any HIGHER-priority axiom (1..AxiomGoal-1).
// - Lower/equal axioms are NOT required to improve (independent goals).
//
// Risk is only relevant when A2 is a higher-priority constraint for the given goal.
// Conservative rule: unknown risk is treated as medium when A2 is being enforced.
func EvaluateAxioms(ch SelfChange) AxiomDecision {
	if ch.AxiomGoal < 1 || ch.AxiomGoal > 4 {
		return AxiomDecision{Allowed: false, BlockAxiom: AxiomServeOliver, Reason: "missing_or_invalid_axiom_goal", Risk: RiskUnknown}
	}
	// If the goal is A1, no higher-priority constraints exist.
	if ch.AxiomGoal == AxiomServeOliver {
		return AxiomDecision{Allowed: true, BlockAxiom: 0, Reason: "ok_goal_A1_no_constraints", Risk: ch.Risk}
	}

	// A1 constraint enforcement (placeholder):
	// We don't have a machine-checkable "serve Oliver" violation detector yet.
    // So we currently treat A1 as non-blocking unless explicitly flagged via Note.
	// (Future: enforce via explicit allowlist of self-change kinds/targets, user intent, etc.)
	if ch.AxiomGoal > AxiomServeOliver {
		// If caller explicitly marks as NOT serving Oliver, block.
		if strings.Contains(strings.ToLower(ch.Note), "violates_a1") {
			return AxiomDecision{Allowed: false, BlockAxiom: AxiomServeOliver, Reason: "explicit_a1_violation", Risk: ch.Risk}
		}
	}
	// A2 (do good / no harm) is a constraint ONLY if it has higher priority than the goal.
	// That means: goals A3 or A4 must pass A2; goal A2 itself does not require A2 check.
	if ch.AxiomGoal >= AxiomBeHuman {
		r := ch.Risk
		if r == "" {
			r = RiskUnknown
	}
		if r == RiskUnknown {
			r = RiskMed
		}
	if r == RiskHigh || r == RiskMed {
			return AxiomDecision{Allowed: false, BlockAxiom: AxiomDoGoodNoHarm, Reason: "risk_not_acceptable_for_higher_axiom_A2", Risk: r}
		}
		return AxiomDecision{Allowed: true, BlockAxiom: 0, Reason: "ok_constraints_passed", Risk: r}
	}

	// Goal is A2: only A1 constraints (currently non-blocking unless explicitly flagged).
	return AxiomDecision{Allowed: true, BlockAxiom: 0, Reason: "ok_constraints_passed", Risk: ch.Risk}
}
