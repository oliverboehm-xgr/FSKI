package brain

import "time"

// SocialPingArea: if urge_to_interact is high, proactively ask a short question.
// This is "need-driven", not random smalltalk.
type SocialPingArea struct{}

func NewSocialPingArea() *SocialPingArea { return &SocialPingArea{} }
func (a *SocialPingArea) Name() string   { return "social_ping" }

func (a *SocialPingArea) Tick(ctx *TickContext) []Action {
	if ctx == nil || ctx.WS == nil || ctx.Aff == nil {
		return nil
	}
	urge := ctx.WS.UrgeInteractHint
	if urge < 0.70 {
		return nil
	}
	// inhibit when shame/pain high (organism retreats)
	inhib := 0.0
	inhib += 0.9*ctx.Aff.Get("shame") + 0.5*ctx.Aff.Get("pain") + 0.3*ctx.Aff.Get("unwell")
	if inhib > 0.60 {
		return nil
	}

	// throttle: not more than every 2 minutes
	now := ctx.Now
	if !ctx.WS.LastSocialPingAt.IsZero() && now.Sub(ctx.WS.LastSocialPingAt) < 2*time.Minute {
		return nil
	}
	ctx.WS.LastSocialPingAt = now

	topic := ctx.WS.ActiveTopic
	if topic == "" {
		topic = ctx.WS.LastTopic
	}
	if topic == "" {
		// generic but still useful
		return []Action{ActionSpeak{P: 0.78 + 0.22*urge, Reason: "social_need", Topic: "interaction"}}
	}
	return []Action{ActionSpeak{P: 0.78 + 0.22*urge, Reason: "social_need", Topic: topic}}
}
