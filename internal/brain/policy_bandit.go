package brain

import (
	"database/sql"
	"math"
	"math/rand"
	"strings"
	"time"
)

var DefaultPolicyActions = []string{
	"direct_answer",
	"ask_clarify",
	"research_then_answer",
	"stance_then_answer",
	"social_ping",
}

type PolicyChoice struct {
	ContextKey string
	Action     string
	Style      string
}

func MakePolicyContext(intentMode string, survival float64, socialCraving float64) string {
	im := strings.ToUpper(strings.TrimSpace(intentMode))
	if im == "" {
		im = "UNKNOWN"
	}
	sv := "sv_lo"
	if survival >= 0.65 {
		sv = "sv_hi"
	}
	sc := "soc_lo"
	if socialCraving >= 0.65 {
		sc = "soc_hi"
	}
	return im + "|" + sv + "|" + sc
}

func sampleBeta(alpha, beta float64) float64 {
	if alpha <= 0 {
		alpha = 1
	}
	if beta <= 0 {
		beta = 1
	}
	x := sampleGamma(alpha)
	y := sampleGamma(beta)
	if x+y == 0 {
		return 0.5
	}
	return x / (x + y)
}

func sampleGamma(k float64) float64 {
	if k < 1 {
		u := rand.Float64()
		return sampleGamma(k+1) * math.Pow(u, 1.0/k)
	}
	d := k - 1.0/3.0
	c := 1.0 / math.Sqrt(9*d)
	for {
		x := rand.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rand.Float64()
		if u < 1-0.0331*(x*x)*(x*x) {
			return d * v
		}
		if math.Log(u) < 0.5*x*x+d*(1-v+math.Log(v)) {
			return d * v
		}
	}
}

func ensureStat(db *sql.DB, ctx, action string) (a, b float64) {
	a, b = 1.0, 1.0
	_ = db.QueryRow(`SELECT alpha,beta FROM policy_stats WHERE context_key=? AND action=?`, ctx, action).Scan(&a, &b)
	if a == 0 && b == 0 {
		_, _ = db.Exec(`INSERT OR IGNORE INTO policy_stats(context_key,action,alpha,beta,updated_at) VALUES(?,?,?,?,?)`,
			ctx, action, 1.0, 1.0, time.Now().Format(time.RFC3339))
		a, b = 1.0, 1.0
	}
	if a < 0.1 {
		a = 0.1
	}
	if b < 0.1 {
		b = 0.1
	}
	return
}

func ChoosePolicy(db *sql.DB, ctx string) PolicyChoice {
	rand.Seed(time.Now().UnixNano())
	bestA := ""
	bestS := -1.0
	for _, act := range DefaultPolicyActions {
		a, b := ensureStat(db, ctx, act)
		s := sampleBeta(a, b)
		if s > bestS {
			bestS = s
			bestA = act
		}
	}
	if bestA == "" {
		bestA = "direct_answer"
	}
	style := "direct"
	if strings.Contains(ctx, "soc_hi") {
		style = "warm"
	}
	if strings.Contains(ctx, "sv_hi") {
		style = "concise"
	}
	return PolicyChoice{ContextKey: ctx, Action: bestA, Style: style}
}

func UpdatePolicy(db *sql.DB, ctx, action string, reward01 float64) {
	if db == nil || ctx == "" || action == "" {
		return
	}
	if reward01 < 0 {
		reward01 = 0
	}
	if reward01 > 1 {
		reward01 = 1
	}
	a, b := ensureStat(db, ctx, action)
	a += reward01
	b += (1.0 - reward01)
	_, _ = db.Exec(`INSERT INTO policy_stats(context_key,action,alpha,beta,updated_at) VALUES(?,?,?,?,?)
		ON CONFLICT(context_key,action) DO UPDATE SET alpha=excluded.alpha, beta=excluded.beta, updated_at=excluded.updated_at`,
		ctx, action, a, b, time.Now().Format(time.RFC3339))
}
