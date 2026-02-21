package main

import (
	"database/sql"
	"encoding/json"
	"net/url"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
	"frankenstein-v0/internal/websense"
)

func answerWithStance(db *sql.DB, oc *ollama.Client, model string, _ *BodyState, _ *brain.AffectState, ws *brain.Workspace, _ *brain.Traits, eg *epi.Epigenome, userText string) (string, error) {
	topic := ""
	if ws != nil && ws.ActiveTopic != "" {
		topic = ws.ActiveTopic
	} else if ws != nil {
		topic = ws.LastTopic
	}
	if strings.TrimSpace(topic) == "" {
		topic = brain.ExtractTopic(userText)
	}
	if strings.TrimSpace(topic) == "" {
		topic = "topic"
	}

	halfLife, minConf, _ := eg.StanceParams()
	if st, ok := brain.GetStance(db, topic); ok && brain.StanceConfidenceDecayed(st) >= minConf {
		return formatStanceReply(st), nil
	}

	results, err := websense.Search(brain.NormalizeSearchQuery(userText), 8)
	if err != nil || len(results) == 0 {
		st := brain.Stance{Topic: topic, Position: 0, Label: "unsicher", Rationale: "Ich habe gerade keine Quellen, um eine fundierte Haltung zu bilden.", Confidence: 0.2, HalfLifeDays: halfLife, UpdatedAt: time.Now()}
		brain.SaveStance(db, st)
		return formatStanceReply(st), nil
	}

	type ev struct {
		URL     string `json:"url"`
		Domain  string `json:"domain"`
		Title   string `json:"title"`
		Snippet string `json:"snippet"`
	}
	evs := make([]ev, 0, 4)
	for i := 0; i < len(results) && i < 4; i++ {
		dom := ""
		if pu, e := url.Parse(results[i].URL); e == nil {
			dom = pu.Hostname()
		}
		evs = append(evs, ev{URL: results[i].URL, Domain: dom, Title: results[i].Title, Snippet: results[i].Snippet})
	}
	evJSON, _ := json.MarshalIndent(evs, "", "  ")
	valJSON, _ := json.MarshalIndent(eg.Values(), "", "  ")

	sys := `Du bist Bunny-StanceEngine.
Du sollst eine Haltung (stance) zum TOPIC bilden.
Eingaben: VALUES (Gewichte), EVIDENCE (Snippets).
Regeln:
- Keine erfundenen Fakten.
- Ergebnis als JSON:
{"position":-1..1,"label":"kurz","rationale":"3-6 bullets","confidence":0..1}`
	user := "TOPIC: " + topic + "\n\nVALUES:\n" + string(valJSON) + "\n\nEVIDENCE:\n" + string(evJSON)
	out, err := oc.Chat(model, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
	if err != nil {
		return "", err
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return "", nil
	}

	var parsed struct {
		Position   float64 `json:"position"`
		Label      string  `json:"label"`
		Rationale  string  `json:"rationale"`
		Confidence float64 `json:"confidence"`
	}
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		st := brain.Stance{Topic: topic, Position: 0, Label: "unsicher", Rationale: "Ich konnte aus den Quellen keine saubere Haltung formen.", Confidence: 0.25, HalfLifeDays: halfLife, UpdatedAt: time.Now()}
		brain.SaveStance(db, st)
		return formatStanceReply(st), nil
	}

	st := brain.Stance{Topic: topic, Position: parsed.Position, Label: strings.TrimSpace(parsed.Label), Rationale: strings.TrimSpace(parsed.Rationale), Confidence: brain.Clamp01(parsed.Confidence), HalfLifeDays: halfLife, UpdatedAt: time.Now()}
	brain.SaveStance(db, st)
	for _, e := range evs {
		brain.AddStanceSource(db, topic, e.URL, e.Domain, e.Snippet, time.Now().Format(time.RFC3339))
	}
	return formatStanceReply(st), nil
}

func formatStanceReply(s brain.Stance) string {
	line1 := "Meine Haltung zu " + s.Topic + ": " + s.Label
	if s.Position <= -0.25 {
		line1 += " (eher dagegen)"
	} else if s.Position >= 0.25 {
		line1 += " (eher dafür)"
	} else {
		line1 += " (ambivalent)"
	}
	line2 := "Begründung: " + s.Rationale
	line3 := "Sicherheit: " + strings.TrimRight(strings.TrimRight(strconv.FormatFloat(s.Confidence, 'f', 2, 64), "0"), ".")
	return line1 + "\n" + line2 + "\n" + line3
}
