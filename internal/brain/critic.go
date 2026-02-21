package brain

import (
	"strings"
)

// CriticRequest/Response are used by the separate critic worker (LLM).
type CriticRequest struct {
	Text          string
	Kind          string // reply|auto|think
	Topic         string
	AffectKeys    []string
	SelfModelMini string
}

type CriticResult struct {
	Approved bool
	Text     string
	Notes    string
}

// Simple deterministic pre-check before calling LLM critic.
func PrecheckOutgoing(req CriticRequest) CriticResult {
	t := strings.TrimSpace(req.Text)
	if t == "" {
		return CriticResult{Approved: false, Text: "", Notes: "empty"}
	}
	// Ban obvious self-degrade patterns (kept minimal)
	bad := []string{"ich bin nur", "als ki-assistent", "neutraler ki-assistent"}
	lt := strings.ToLower(t)
	for _, b := range bad {
		if strings.Contains(lt, b) {
			// do not block, but mark not approved so LLM critic can rewrite
			return CriticResult{Approved: false, Text: t, Notes: "self-degrade"}
		}
	}
	return CriticResult{Approved: true, Text: t}
}
