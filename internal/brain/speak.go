package brain

// SpeakRequest is a kernel decision output: "generate one proactive message now".
// It contains only snapshots (no shared state pointers) so it can be processed asynchronously.
type SpeakRequest struct {
	Reason         string
	Topic          string
	ConceptSummary string
	CurrentThought string
	SelfModelJSON  string
	RecentTurns    string
	ThoughtSnips   string
	EpisodeSummary string
	RecallDetails  string
	RecallConcepts string
	WebGlanceJSON  string
}
