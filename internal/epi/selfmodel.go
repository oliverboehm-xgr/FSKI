package epi

import "time"

type SelfModel struct {
	Body struct {
		Energy       float64 `json:"energy"`
		WebCountHour int     `json:"webCountHour"`
		Cooldown     string  `json:"cooldownUntil"`
	} `json:"body"`
	Epigenome struct {
		EnabledModules []string `json:"enabledModules"`
		Version        int      `json:"version"`
	} `json:"epigenome"`
}

// BuildSelfModel is intentionally minimal for v0.1.
// Later: add runtime stats (mem/cpu/disk), message queue, interests, etc.
func BuildSelfModel(body any, eg *Epigenome) *SelfModel {
	// body is passed as interface to avoid importing main package types.
	// We only fill what we can without reflection here; main already formats core state in prompts.
	sm := &SelfModel{}
	sm.Epigenome.EnabledModules = eg.EnabledModuleNames()
	sm.Epigenome.Version = eg.Version

	// Note: energy/cooldown get embedded via main's prompt formatting in v0.1.
	// Next patch: make BodyState a shared type or move it to internal/runtime.
	sm.Body.Cooldown = time.Time{}.Format(time.RFC3339)
	return sm
}
