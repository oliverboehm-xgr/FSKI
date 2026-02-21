package brain

import (
	"time"
)

// Human-like daydream trigger (images + inner speech produced by a separate worker).
type DaydreamArea struct{}

func NewDaydreamArea() *DaydreamArea { return &DaydreamArea{} }
func (a *DaydreamArea) Name() string { return "daydream" }

func (a *DaydreamArea) Tick(ctx *TickContext) []Action {
	if ctx == nil || ctx.EG == nil || ctx.WS == nil || ctx.Dr == nil {
		return nil
	}
	intervalSec, minCur, minEnergy, _, enabled := ctx.EG.DaydreamParams()
	if !enabled {
		return nil
	}
	if ctx.Dr.Curiosity < minCur {
		return nil
	}

	// avoid daydreaming if ashamed/painful
	inhib := 0.0
	if ctx.Aff != nil {
		inhib += 0.8*ctx.Aff.Get("shame") + 0.4*ctx.Aff.Get("pain") + 0.3*ctx.Aff.Get("unwell")
	}
	if inhib > 0.55 {
		return nil
	}

	// body is opaque here; we use Workspace energy hint if present
	if ctx.WS.EnergyHint > 0 && ctx.WS.EnergyHint < minEnergy {
		return nil
	}

	now := ctx.Now
	if !ctx.WS.LastDaydreamAt.IsZero() && now.Sub(ctx.WS.LastDaydreamAt) < time.Duration(intervalSec)*time.Second {
		return nil
	}

	// trigger
	ctx.WS.LastDaydreamAt = now
	return []Action{ActionDaydream{P: 0.6 + 0.4*ctx.Dr.Curiosity}}
}
