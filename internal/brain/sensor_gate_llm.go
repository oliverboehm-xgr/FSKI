package brain

import (
	"encoding/json"
	"errors"
	"strings"

	"frankenstein-v0/internal/ollama"
)

type llmGateOut struct {
	NeedWeb    bool    `json:"need_web"`
	Confidence float64 `json:"confidence"`
	Query      string  `json:"query"`
	Reason     string  `json:"reason"`
}

func extractJSONObject(s string) (string, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", false
	}
	// strip code fences
	if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimPrefix(s, "```JSON")
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSpace(s)
		if i := strings.LastIndex(s, "```"); i >= 0 {
			s = strings.TrimSpace(s[:i])
		}
	}
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start < 0 || end < 0 || end <= start {
		return "", false
	}
	return strings.TrimSpace(s[start : end+1]), true
}

// CortexWebGate asks a small LLM to decide whether WebSense is needed.
// IMPORTANT: If uncertain, it must prefer need_web=true (avoid hallucinations).
func CortexWebGate(oc *ollama.Client, model string, userText string, intent Intent, ws *Workspace) (need bool, conf float64, query string, reason string, err error) {
	if oc == nil || strings.TrimSpace(model) == "" {
		return false, 0, "", "", errors.New("ollama_missing")
	}
	if ws != nil && ws.TrainingDryRun {
		return false, 0, "", "", errors.New("dry_run")
	}

	sys := "Du bist ein Sensor-Gate. Du beantwortest NICHT die Nutzerfrage. " +
		"Du entscheidest nur, ob ein WebSense-Aufruf nötig ist, um Halluzinationen zu vermeiden. " +
		"Wenn du unsicher bist: need_web=true. " +
		"Output ONLY JSON: {\"need_web\":bool,\"confidence\":0..1,\"query\":string,\"reason\":string}."

	webAllowed := true
	survivalMode := false
	if ws != nil {
		webAllowed = ws.WebAllowed
		survivalMode = ws.SurvivalMode
	}

	user := "USER_TEXT:\n" + userText +
		"\n\nINTENT_MODE:" + IntentToMode(intent) +
		"\nWEB_ALLOWED:" + boolTo01(webAllowed) +
		"\nSURVIVAL_MODE:" + boolTo01(survivalMode) +
		"\n\nEntscheide need_web. Wenn need_web=true, gib eine kurze Suchquery (Deutsch)."

	out, e := oc.Chat(model, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
	if e != nil {
		return false, 0, "", "", e
	}
	js, ok := extractJSONObject(out)
	if !ok {
		return false, 0, "", "", errors.New("gate_non_json")
	}
	var g llmGateOut
	if err := json.Unmarshal([]byte(js), &g); err != nil {
		return false, 0, "", "", err
	}
	conf = g.Confidence
	if conf < 0 {
		conf = 0
	}
	if conf > 1 {
		conf = 1
	}
	need = g.NeedWeb
	query = strings.TrimSpace(g.Query)
	reason = strings.TrimSpace(g.Reason)
	return
}

func boolTo01(b bool) string {
	if b {
		return "1"
	}
	return "0"
}
