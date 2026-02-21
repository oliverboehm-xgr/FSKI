package brain

// Speak trigger is separated from actual text generation.
type SpeakArea struct{}

func NewSpeakArea() *SpeakArea    { return &SpeakArea{} }
func (a *SpeakArea) Name() string { return "speak" }

func (a *SpeakArea) Tick(ctx *TickContext) []Action {
	if ctx == nil || ctx.WS == nil || ctx.Dr == nil {
		return nil
	}
	// If we just generated a salient thought, urge grows elsewhere; here we only decide to speak.
	// The caller will apply cooldown and generate actual message via speaker worker.
	urge := ctx.Dr.UrgeToShare
	inhib := 0.0
	if ctx.Aff != nil {
		inhib += 0.9*ctx.Aff.Get("shame") + 0.4*ctx.Aff.Get("unwell") + 0.3*ctx.Aff.Get("pain") + 0.3*ctx.Aff.Get("fear")
	}
	thr := 0.75 + 0.20*inhib
	if urge < thr {
		return nil
	}
	topic := ctx.WS.ActiveTopic
	if topic == "" {
		topic = ctx.WS.LastTopic
	}
	if topic == "" {
		return nil
	}
	return []Action{ActionSpeak{P: 0.8 + 0.2*urge, Reason: "urge_to_share high", Topic: topic}}
}
