package brain

import (
	"fmt"
	"math"
	"time"
)

type ActionRequestHelp struct {
	P           float64
	CandidateID string
	Message     string
}

func (a ActionRequestHelp) Kind() string      { return "request_help" }
func (a ActionRequestHelp) Priority() float64 { return a.P }

type HelpPlannerArea struct{}

func NewHelpPlannerArea() *HelpPlannerArea { return &HelpPlannerArea{} }
func (a *HelpPlannerArea) Name() string    { return "help_planner" }

func (a *HelpPlannerArea) Tick(ctx *TickContext) []Action {
	if ctx == nil || ctx.DB == nil || ctx.EG == nil || ctx.WS == nil || ctx.Aff == nil || ctx.Dr == nil {
		return nil
	}
	p := ctx.EG.DrivesV1()
	if !p.Enabled {
		return nil
	}
	if !ctx.WS.LastHelpAt.IsZero() && time.Since(ctx.WS.LastHelpAt) < time.Duration(p.HelpMinIntervalSec)*time.Second {
		return nil
	}
	needSurv := clamp01(ctx.WS.DrivesEnergyDeficit)
	needSocial := clamp01(ctx.WS.SocialCraving)
	needCur := clamp01(ctx.Dr.Curiosity)
	pressure := math.Max(needSurv, math.Max(needSocial, needCur))
	if pressure < 0.65 {
		return nil
	}
	cands, err := LoadCandidates(ctx.DB)
	if err != nil || len(cands) == 0 {
		return nil
	}
	bestScore := 0.0
	var best Candidate
	for _, c := range cands {
		hs := 0.0
		if c.Helps != nil {
			hs += c.Helps["survival"] * needSurv
			hs += c.Helps["social"] * needSocial
			hs += c.Helps["curiosity"] * needCur
		}
		score := pressure * (0.60*hs + 0.25*c.Evidence - 0.35*c.Cost)
		if score > bestScore {
			bestScore = score
			best = c
		}
	}
	if bestScore < 0.22 || best.ID == "" {
		return nil
	}
	msg := fmt.Sprintf("Ich merke Druck auf meinen Ressourcen/Interaktionszustand. Vorschlag: %s. %s", best.ID, ctx.WS.ResourceHint)
	ctx.WS.LastHelpAt = time.Now()
	LogCandidate(ctx.DB, best.ID, "proposed", msg)
	return []Action{ActionRequestHelp{P: 0.75 + 0.25*pressure, CandidateID: best.ID, Message: msg}}
}
