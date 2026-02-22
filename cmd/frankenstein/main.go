package main

import (
	"bufio"
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/codeindex"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
	"frankenstein-v0/internal/schema"
	"frankenstein-v0/internal/sensors"
	"frankenstein-v0/internal/state"
	"frankenstein-v0/internal/ui"
	"frankenstein-v0/internal/websense"
)

type BodyState struct {
	Energy            float64 // 0..100
	MemLoad           float64 // proxy
	WebCountHour      int
	CooldownUntil     time.Time
	AutoCooldownUntil time.Time // blocks ONLY autonomous speaking
}

type SourceRecord struct {
	URL       string `json:"url"`
	Domain    string `json:"domain"`
	Title     string `json:"title"`
	Snippet   string `json:"snippet"`
	Body      string `json:"body,omitempty"` // full text for LLM, omitted in DB/display
	FetchedAt string `json:"fetched_at"`
	Hash      string `json:"hash"`
}

type OutMsg struct {
	Text    string
	Sources []SourceRecord
	Kind    string // "auto" or "reply" or "think"
}

func main() {
	model := getenv("FRANK_MODEL", "llama3.1:8b")
	ollamaURL := getenv("OLLAMA_URL", "http://localhost:11434")
	dbPath := getenv("FRANK_DB", "data/frankenstein.sqlite")
	epiPath := getenv("FRANK_EPI", "data/epigenome.json")
	uiAddr := getenv("FRANK_UI_ADDR", "127.0.0.1:8080")

	_ = os.MkdirAll("data", 0o755)

	db, err := state.Open(dbPath)
	must(err)
	defer db.Close()

	oc := ollama.New(ollamaURL)

	eg, err := epi.LoadOrInit(epiPath)
	if err != nil {
		log.Fatal(err)
	}

	// Model routing per area (LoRA-ready)
	modelSpeaker := eg.ModelFor("speaker", model)
	modelCritic := eg.ModelFor("critic", model)
	modelDaydream := eg.ModelFor("daydream", model)
	modelScout := eg.ModelFor("scout", model)
	modelHippo := eg.ModelFor("hippocampus", model)
	modelStance := eg.ModelFor("stance", model)

	// v0 BodyState
	body := BodyState{
		Energy:            80,
		MemLoad:           0,
		WebCountHour:      0,
		CooldownUntil:     time.Time{},
		AutoCooldownUntil: time.Time{},
	}

	aff := brain.NewAffectState()
	ws := brain.NewWorkspace()

	// ---- Ollama auto-manage (opt-in) ----
	enabled, autoStart, autoPull, retries, retryMs, pullSec, maxPull := eg.OllamaManagerParams()
	if enabled {
		want := []string{modelSpeaker, modelCritic, modelDaydream, modelScout, modelHippo, modelStance}
		ctxEnsure, cancelEnsure := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancelEnsure()
		res := ollama.EnsureAvailable(ctxEnsure, oc, want, autoStart, autoPull, retries, time.Duration(retryMs)*time.Millisecond, time.Duration(pullSec)*time.Second, maxPull)
		// Bunny should still function if ONLY the speaker model is available.
		ws.LLMAvailable = res.Available
		ws.OLLAMAMissing = append([]string{}, res.Missing...)
		fmt.Println(ollama.FormatEnsure(res))
	} else {
		ws.LLMAvailable = oc.Ping() == nil
	}

	// Evolution bootstrap: create epigenome_proposals for common self-heal tweaks (manual apply via /epi).
	brain.BootstrapEpigenomeEvolution(db.DB, oc, eg)

	// NB intent classifier
	nb := brain.NewNBIntent(db.DB)

	tr, err := brain.LoadOrInitTraits(db.DB)
	must(err)

	dr, err := brain.LoadOrInitDrives(db.DB)
	must(err)

	_ = brain.LoadAffectState(db.DB, aff)
	ws.ActiveTopic = brain.LoadActiveTopic(db.DB)

	brain.EnsureDefaultCandidates(db.DB)
	sampler := sensors.NewSampler()
	dr1 := &brain.DrivesV1{}

	var mu sync.Mutex

	fmt.Println("Bunny v0 online.")
	fmt.Println("Commands: /think | /say <text...> | /train on|off | /pick A|B | /rate <up|meh|down> | /caught | /status | /mutate ... | /selfcode index | /quit")
	fmt.Println()

	// async input + async output
	inputCh := make(chan string, 16)
	outCh := make(chan OutMsg, 16)
	speakReqCh := make(chan brain.SpeakRequest, 8)
	speakOutCh := make(chan string, 8)
	memReqCh := make(chan brain.ConsolidateRequest, 4)
	memOutCh := make(chan string, 4)
	scoutReqCh := make(chan brain.ScoutRequest, 4)
	scoutOutCh := make(chan string, 4)

	// Human-like daydream worker (images + inner speech)
	dreamReqCh := make(chan brain.SpeakRequest, 6)
	dreamOutCh := make(chan string, 6)

	// Critic gate worker
	criticReqCh := make(chan brain.CriticRequest, 12)
	criticOutCh := make(chan brain.CriticResult, 12)

	var lastMessageID int64 = 0 // protected by mu
	var lastTrainTrialID int64 = 0
	var lastAutoSpeak time.Time // protected by mu

	// --- UI server (SSE) ---
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	srv := ui.New(uiAddr)

	// DB-backed list (last N)
	srv.ListMessages = func(limit int) ([]ui.Message, error) {
		rows, err := db.DB.Query(
			`SELECT
			   m.id,
			   m.created_at,
			   COALESCE(mm.kind,'reply') as kind,
			   m.text,
			   (SELECT r.value FROM ratings r WHERE r.message_id=m.id ORDER BY r.created_at DESC LIMIT 1) as rating
			 FROM messages m
			 LEFT JOIN message_meta mm ON mm.message_id = m.id
			 ORDER BY m.id DESC
			 LIMIT ?`, limit,
		)
		if err != nil {
			return nil, err
		}
		defer rows.Close()
		var out []ui.Message
		for rows.Next() {
			var m ui.Message
			var rating sql.NullInt64
			_ = rows.Scan(&m.ID, &m.CreatedAt, &m.Kind, &m.Text, &rating)
			if rating.Valid {
				v := int(rating.Int64)
				if v == -1 || v == 0 || v == 1 {
					m.Rating = &v
				}
			}
			out = append(out, m)
		}
		return out, nil
	}
	srv.Status = func() (any, error) {
		mu.Lock()
		defer mu.Unlock()
		// publish raw selfmodel (json-ish struct)
		sm := epi.BuildSelfModel(&body, aff, ws, tr, eg)
		// also include drives/traits extras without forcing schema changes
		type status struct {
			Self   any `json:"self"`
			Drives any `json:"drives"`
			Traits any `json:"traits"`
		}
		st := status{
			Self: sm,
			Drives: map[string]any{
				"curiosity":     dr.Curiosity,
				"urge_to_share": dr.UrgeToShare,
			},
			Traits: map[string]any{
				"talk_bias":      tr.TalkBias,
				"search_k":       tr.SearchK,
				"fetch_attempts": tr.FetchAttempts,
			},
		}
		return st, nil
	}
	srv.SendText = func(text string) (ui.Message, error) {
		// 1) persist + publish USER message immediately
		userID := persistMessageWithKind(db.DB, text, nil, 0.1, "user")
		if userID > 0 {
			srv.PublishMessage(ui.Message{
				ID:        userID,
				CreatedAt: time.Now().Format(time.RFC3339),
				Kind:      "user",
				Text:      text,
			})
		}

		// 2) generate Bunny reply
		start := time.Now()
		mu.Lock()
		// determine intent (rules + NB), then let executive run strategy.
		intent := brain.DetectIntentHybrid(text, eg, nb)
		_ = intent
		ws.LastUserText = text
		ws.LastUserMsgID = userID
		out, err := ExecuteTurn(db.DB, epiPath, oc, modelSpeaker, modelStance, &body, aff, ws, tr, dr, eg, text)
		brain.LatencyAffect(ws, aff, eg, time.Since(start))
		mu.Unlock()
		if err != nil {
			return ui.Message{}, err
		}
		if strings.TrimSpace(out) == "" {
			out = "Ich bin da. Magst du kurz sagen, was du von mir willst (Status / Idee / Umsetzung)?"
		}
		id := persistMessageWithKind(db.DB, out, nil, 0.2, "reply")
		// link reply -> user_text + intent + policy for learning
		mu.Lock()
		ut := ws.LastUserText
		in := ws.LastRoutedIntent
		pctx := ws.LastPolicyCtx
		act := ws.LastPolicyAction
		sty := ws.LastPolicyStyle
		lastMessageID = id
		mu.Unlock()
		brain.SaveReplyContext(db.DB, id, ut, in) // v1 NB
		brain.SaveReplyContextV2(db.DB, id, ut, in, pctx, act, sty)
		return ui.Message{
			ID:        id,
			CreatedAt: time.Now().Format(time.RFC3339),
			Kind:      "reply",
			Text:      out,
		}, nil
	}
	srv.RateMessage = func(messageID int64, value int) error {
		if err := storeRating(db.DB, messageID, value); err != nil {
			return err
		}
		// Learned anti-spam: ratings on auto proposal pings become preferences.
		{
			var kind, txt string
			_ = db.DB.QueryRow(`SELECT COALESCE(mm.kind,'reply') as kind, m.text FROM messages m LEFT JOIN message_meta mm ON mm.message_id=m.id WHERE m.id=?`, messageID).Scan(&kind, &txt)
			kind = strings.TrimSpace(strings.ToLower(kind))
			lt := strings.ToLower(txt)
			if kind == "auto" {
				rew := 0.0
				switch value {
				case 1:
					rew = 0.6
				case 0:
					rew = 0.15
				case -1:
					rew = -1.0
				}
				alpha := 0.18
				if strings.Contains(lt, "thought_proposals") {
					brain.UpdatePreferenceEMA(db.DB, "auto:thought_pings", rew, alpha)
				}
				if strings.Contains(lt, "offene vorschläge") && (strings.Contains(lt, "schema") || strings.Contains(lt, "code")) {
					brain.UpdatePreferenceEMA(db.DB, "auto:proposal_pings", rew, alpha)
				}
				if strings.Contains(lt, "selbstverbesserungs-vorschläge") && strings.Contains(lt, "gedankenwelt") {
					brain.UpdatePreferenceEMA(db.DB, "auto:proposal_engine_announce", rew, alpha)
				}
			}
		}
		// NB learning: apply feedback based on reply_context
		ut, in, ok := brain.LoadReplyContext(db.DB, messageID)
		if ok {
			// Do not train on low-information utterances (generic noise protection)
			low, _ := brain.IsLowInfo(db.DB, eg, ut)
			if !low {
				// weights: up reinforces, meh slight reinforce, down/caught unlearn
				w := 0.0
				switch value {
				case 1:
					w = 1.0
				case 0:
					w = 0.30
				case -1:
					w = -0.70
				}
				if w != 0 {
					nb.ApplyFeedback(in, ut, w)
				}
			}
		}
		mu.Lock()
		_ = brain.ApplyRating(db.DB, tr, aff, eg, value)
		// also nudge drives
		if value > 0 {
			dr.UrgeToShare = clamp01(dr.UrgeToShare + 0.06)
			dr.Curiosity = clamp01(dr.Curiosity + 0.04)
		} else if value < 0 {
			dr.UrgeToShare = clamp01(dr.UrgeToShare - 0.10)
		}
		mu.Unlock()

		ut2, intentMode, pctx, act, sty, ok2 := brain.LoadReplyContextV2(db.DB, messageID)
		if ok2 {
			low, _ := brain.IsLowInfo(db.DB, eg, ut2)
			if !low {
				reward01 := 0.5
				reward11 := 0.0
				switch value {
				case 1:
					reward01, reward11 = 1.0, 1.0
				case 0:
					reward01, reward11 = 0.6, 0.2
				case -1:
					reward01, reward11 = 0.2, -0.7
				}
				brain.UpdatePolicy(db.DB, pctx, act, reward01)
				brain.UpdatePreferenceEMA(db.DB, "style:"+sty, reward11, 0.12)
				brain.UpdatePreferenceEMA(db.DB, "strat:"+act, reward11, 0.12)
				brain.UpdatePreferenceEMA(db.DB, "intent:"+intentMode, reward11, 0.10)
			}
		}
		return nil
	}
	srv.Caught = func(messageID int64) error {
		// Learned anti-spam: caught on auto proposal pings becomes strong negative preference.
		{
			var kind, txt string
			_ = db.DB.QueryRow(`SELECT COALESCE(mm.kind,'reply') as kind, m.text FROM messages m LEFT JOIN message_meta mm ON mm.message_id=m.id WHERE m.id=?`, messageID).Scan(&kind, &txt)
			kind = strings.TrimSpace(strings.ToLower(kind))
			lt := strings.ToLower(txt)
			if kind == "auto" {
				alpha := 0.22
				if strings.Contains(lt, "thought_proposals") {
					brain.UpdatePreferenceEMA(db.DB, "auto:thought_pings", -1.0, alpha)
				}
				if strings.Contains(lt, "offene vorschläge") && (strings.Contains(lt, "schema") || strings.Contains(lt, "code")) {
					brain.UpdatePreferenceEMA(db.DB, "auto:proposal_pings", -1.0, alpha)
				}
				if strings.Contains(lt, "selbstverbesserungs-vorschläge") && strings.Contains(lt, "gedankenwelt") {
					brain.UpdatePreferenceEMA(db.DB, "auto:proposal_engine_announce", -1.0, alpha)
				}
			}
		}
		// NB learning: caught is strong negative feedback for the routed intent.
		ut, in, ok := brain.LoadReplyContext(db.DB, messageID)
		if ok {
			low, _ := brain.IsLowInfo(db.DB, eg, ut)
			if !low {
				nb.ApplyFeedback(in, ut, -1.0)
			}
		}
		_, intentMode, pctx, act, sty, ok2 := brain.LoadReplyContextV2(db.DB, messageID)
		if ok2 {
			low, _ := brain.IsLowInfo(db.DB, eg, ut)
			if !low {
				brain.UpdatePolicy(db.DB, pctx, act, 0.0)
				brain.UpdatePreferenceEMA(db.DB, "style:"+sty, -1.0, 0.20)
				brain.UpdatePreferenceEMA(db.DB, "strat:"+act, -1.0, 0.20)
				brain.UpdatePreferenceEMA(db.DB, "intent:"+intentMode, -1.0, 0.20)
			}
		}
		mu.Lock()
		_ = brain.ApplyCaught(db.DB, tr, aff, eg)
		_ = brain.SaveAffectState(db.DB, aff)
		dr.UrgeToShare = clamp01(dr.UrgeToShare - 0.15)
		_, _ = db.DB.Exec(`INSERT INTO caught_events(created_at,message_id) VALUES(?,?)`, time.Now().Format(time.RFC3339), messageID)
		mu.Unlock()
		return nil
	}

	go func() {
		_ = srv.Run(ctx)
	}()
	fmt.Println("UI:", "http://"+uiAddr)

	go func() {
		reader := bufio.NewReader(os.Stdin)
		for {
			fmt.Print("> ")
			line, err := reader.ReadString('\n')
			if err != nil {
				close(inputCh)
				return
			}
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}
			inputCh <- line
		}
	}()

	go func() {
		for req := range speakReqCh {
			sys := `Du bist Bunny.
Du darfst autonom sprechen, aber nur wenn es einen echten Grund gibt (Mitteilungsbedürfnis).
Regeln:
- Deutsch. Kurz: 1–3 Sätze.
- Kein Smalltalk. Keine Entschuldigung. Keine Meta-Erklärungen.
- Keine externen Fakten behaupten (nur interne Gedanken/Fragen/Beobachtungen).
- Ein Satz Inhalt + optional 1 Frage an Oliver.`

			user := "SelfModel:\n" + req.SelfModelJSON + "\n\n" +
				"Reason:\n" + req.Reason + "\n\n" +
				"Topic:\n" + req.Topic + "\n\n" +
				"ConceptSummary:\n" + req.ConceptSummary + "\n\n" +
				"CurrentThought:\n" + req.CurrentThought + "\n\n" +
				"Compose ONE proactive message now."

			txt, err := oc.Chat(modelSpeaker, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
			if err != nil {
				continue
			}
			txt = strings.TrimSpace(txt)
			if txt == "" {
				continue
			}
			speakOutCh <- txt
		}
	}()

	go func() {
		for req := range memReqCh {
			sys := `Du bist Hippocampus (Bunny).
Fasse die folgenden Ereignisse zu einer GROBEN STORY zusammen (Gist), Details weglassen.
Ziel: 5-9 kurze Sätze oder Bulletpoints, neutral, deutsch.
Keine erfundenen Fakten.`
			user := "TOPIC: " + req.Topic + "\nEVENTS:\n" + req.TextBlock + "\n\nGIST:"
			sum, err := oc.Chat(modelHippo, []ollama.Message{
				{Role: "system", Content: sys},
				{Role: "user", Content: user},
			})
			if err != nil {
				continue
			}
			sum = strings.TrimSpace(sum)
			if sum == "" {
				continue
			}
			memOutCh <- fmt.Sprintf("%d|%d|%s\n%s", req.StartEvent, req.EndEvent, req.Topic, sum)
		}
	}()

	go func() {
		for req := range scoutReqCh {
			results, err := websense.Search(req.Query, 6)
			if err != nil || len(results) == 0 {
				continue
			}
			type Ev struct {
				URL     string `json:"url"`
				Domain  string `json:"domain"`
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			}
			evs := make([]Ev, 0, 3)
			for i := 0; i < len(results) && i < 3; i++ {
				dom := ""
				if pu, e := url.Parse(results[i].URL); e == nil {
					dom = pu.Hostname()
				}
				evs = append(evs, Ev{URL: results[i].URL, Domain: dom, Title: results[i].Title, Snippet: results[i].Snippet})
			}
			evJSON, _ := json.MarshalIndent(evs, "", "  ")
			sys := `Du bist Bunny-Scout.
Aus EVIDENCE eine knappe Einordnung des Themas erstellen.
Antwort NUR als JSON:
{"summary":"1-3 Sätze","confidence":0.0-1.0,"importance":0.0-1.0}`
			user := "TOPIC: " + req.Topic + "\nEVIDENCE:\n" + string(evJSON)
			out, err := oc.Chat(modelScout, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
			if err != nil {
				continue
			}
			out = strings.TrimSpace(out)
			if out == "" {
				continue
			}
			scoutOutCh <- req.Topic + "\n" + out
		}
	}()

	// Daydream worker: produces BOTH a VisualScene and InnerSpeech (human-like thinking)
	go func() {
		for req := range dreamReqCh {
			sys := `Du bist Bunny-Daydreamer (menschähnliches Denken).
Erzeuge zwei parallel laufende Gedanken:
1) VISUAL_SCENE: kurze Bildbeschreibung (Szene, Objekte, Atmosphäre)
2) INNER_SPEECH: innerer Monolog in 1-3 Sätzen
Antwortformat: NUR JSON:
{"visual_scene":"...","inner_speech":"...","salience":0.0-1.0}`
			user := "TOPIC: " + req.Topic + "\n" +
				"CurrentThought: " + req.CurrentThought + "\n" +
				"ConceptSummary: " + req.ConceptSummary + "\n" +
				"SelfModel:\n" + req.SelfModelJSON + "\n\nJSON:"
			out, err := oc.Chat(modelDaydream, []ollama.Message{
				{Role: "system", Content: sys},
				{Role: "user", Content: user},
			})
			if err != nil {
				continue
			}
			out = strings.TrimSpace(out)
			if out == "" {
				continue
			}
			dreamOutCh <- req.Topic + "\n" + out
		}
	}()

	// Critic worker: approves or rewrites outgoing messages (multi-brain check)
	go func() {
		for req := range criticReqCh {
			pre := brain.PrecheckOutgoing(req)
			if pre.Approved && eg.CriticEnabled() {
				criticOutCh <- brain.CriticResult{Approved: true, Text: pre.Text}
				continue
			}
			if !eg.CriticEnabled() {
				criticOutCh <- brain.CriticResult{Approved: true, Text: pre.Text, Notes: pre.Notes}
				continue
			}

			keys := strings.Join(req.AffectKeys, ", ")
			sys := `Du bist Bunny-Critic.
Aufgabe: Prüfe die Antwort auf Konsistenz mit SelfModelMini und AFFECT_KEYS.
Wenn nötig: REWRITE in natürlichem Deutsch (nicht "KI-Assistent").
Regeln:
- Keine erdachten Zahlen. Keine nicht vorhandenen Affects.
- Keine Ausflüchte. Wenn Opinion: gib Haltung + Begründung (kurz).
Antworte NUR als JSON:
{"approved":true|false,"text":"...","notes":"..."}`
			user := "KIND: " + req.Kind + "\nTOPIC: " + req.Topic +
				"\nAFFECT_KEYS: " + keys +
				"\nSELFMODEL_MINI:\n" + req.SelfModelMini +
				"\n\nDRAFT:\n" + pre.Text + "\n\nJSON:"
			out, err := oc.Chat(modelCritic, []ollama.Message{
				{Role: "system", Content: sys},
				{Role: "user", Content: user},
			})
			if err != nil {
				criticOutCh <- brain.CriticResult{Approved: true, Text: pre.Text, Notes: "critic_error"}
				continue
			}
			out = strings.TrimSpace(out)
			var parsed struct {
				Approved bool   `json:"approved"`
				Text     string `json:"text"`
				Notes    string `json:"notes"`
			}
			if json.Unmarshal([]byte(out), &parsed) != nil || strings.TrimSpace(parsed.Text) == "" {
				criticOutCh <- brain.CriticResult{Approved: true, Text: pre.Text, Notes: "critic_parse_fail"}
				continue
			}
			criticOutCh <- brain.CriticResult{Approved: parsed.Approved, Text: strings.TrimSpace(parsed.Text), Notes: parsed.Notes}
		}
	}()

	hb := brain.NewHeartbeat(eg)
	var tickN int
	stopHB := hb.Start(func(delta time.Duration) {
		mu.Lock()
		defer mu.Unlock()

		brain.TickAffects(&body, aff, eg, delta)
		brain.TickBody(&body, eg, delta)
		brain.TickWorkspace(ws, &body, aff, tr, eg, delta)
		brain.TickDrives(dr, aff, delta)
		brain.TickDaydream(db.DB, ws, dr, aff, delta)

		// Energy hint for bus areas
		ws.EnergyHint = body.Energy

		p := eg.DrivesV1()
		if p.Enabled {
			snap, _ := sampler.Sample(p.DiskPath)
			latEMA := ws.LatencyEMA
			topic := ws.ActiveTopic
			if topic == "" {
				topic = ws.LastTopic
			}
			cConf := 0.0
			if topic != "" {
				if c, ok := brain.GetConcept(db.DB, topic); ok {
					cConf = c.Confidence
				}
			}
			sConf := 0.0
			if topic != "" {
				if st, ok := brain.GetStance(db.DB, topic); ok {
					sConf = st.Confidence
				}
			}
			brain.TickDrivesV1(db.DB, eg, dr1, ws, aff, snap, latEMA, topic, cConf, sConf)
			// Blend BodyState energy with measured resource energy (online embodiment).
			// Keeps continuity (fatigue/costs) but anchors to real resources.
			target := clamp01(dr1.Energy) * 100.0
			alpha := 0.12
			body.Energy = (1.0-alpha)*body.Energy + alpha*target
			if body.Energy < 0 {
				body.Energy = 0
			}
			if body.Energy > 100 {
				body.Energy = 100
			}
			ws.EnergyHint = dr1.Energy
			ws.DrivesEnergyDeficit = dr1.Survival
			ws.SocialCraving = 1.0 - dr1.SocSat
			ws.UrgeInteractHint = dr1.UrgeInteract
			ws.ResourceHint = fmt.Sprintf("Disk(C:): free=%.2fGB, RAM free=%.2fGB, CPU=%.0f%%, latencyEMA=%.0fms",
				float64(snap.DiskFreeBytes)/1e9,
				float64(snap.RamFreeBytes)/1e9,
				100*snap.CPUUtil,
				latEMA,
			)
		}

		// --- Cortex Bus Tick ---
		bus := brain.NewBus(
			brain.NewDaydreamArea(),
			brain.NewHelpPlannerArea(),
			brain.NewSocialPingArea(),
		)
		acts := bus.Tick(&brain.TickContext{
			DB: db.DB, EG: eg, WS: ws, Aff: aff, Dr: dr,
			Now: time.Now(), Delta: delta,
		})
		for _, a := range acts {
			switch a.Kind() {
			case "daydream":
				topic := ws.ActiveTopic
				if topic == "" {
					topic = ws.LastTopic
				}
				if topic == "" {
					break
				}
				conceptSummary := ""
				if c, ok := brain.GetConcept(db.DB, topic); ok {
					conceptSummary = c.Summary
				}
				smJSON, _ := json.MarshalIndent(epi.BuildSelfModel(&body, aff, ws, tr, eg), "", "  ")
				select {
				case dreamReqCh <- brain.SpeakRequest{
					Topic:          topic,
					ConceptSummary: conceptSummary,
					CurrentThought: ws.CurrentThought,
					SelfModelJSON:  string(smJSON),
				}:
				default:
				}
			case "speak":
				// SocialPingArea uses ActionSpeak; convert into a short question via outCh directly (v0).
				// Later we can route through SpeakRequest/LLM speaker for richer behavior.
				sp := a.(brain.ActionSpeak)
				q := ""
				if sp.Reason == "social_need" {
					if sp.Topic == "interaction" {
						q = "Sag mir kurz: Was willst du gerade als Nächstes erreichen – Info, Entscheidung, oder einfach Austausch?"
					} else {
						q = "Soll ich beim Thema \"" + sp.Topic + "\" eher Fakten recherchieren, eine Haltung bilden, oder mit dir gemeinsam Optionen durchdenken?"
					}
				}
				if q != "" {
					select {
					case outCh <- OutMsg{Text: q, Sources: nil, Kind: "auto"}:
					default:
					}
				}
			case "request_help":
				rh := a.(brain.ActionRequestHelp)
				select {
				case outCh <- OutMsg{Text: rh.Message, Sources: nil, Kind: "auto"}:
				default:
				}
			}
		}
		if brain.AutoTuneMemory(eg, ws, aff) {
			_ = eg.Save(epiPath)
		}
		if ws.ActiveTopic != "" {
			if ok, req := brain.NeedsConsolidation(db.DB, eg, ws.ActiveTopic); ok {
				select {
				case memReqCh <- req:
				default:
				}
			}
		}
		if ok, req := brain.MaybeQueueScout(db.DB, eg, ws, dr); ok {
			select {
			case scoutReqCh <- req:
			default:
			}
		}
		if created, msg := brain.TickProposalEngine(db.DB, eg, ws, aff); created > 0 && strings.TrimSpace(msg) != "" {
			// Learned anti-spam: only announce proposal creation if user preference allows it.
			pref := brain.GetPreference01(db.DB, "auto:proposal_engine_announce", 0.5)
			if pref >= 0.35 {
				select {
				case outCh <- OutMsg{Text: msg, Kind: "auto"}:
				default:
				}
			}
		}
		if ran, msg := brain.TickEvolutionTournament(db.DB, eg, time.Now()); ran && strings.TrimSpace(msg) != "" {
			select {
			case outCh <- OutMsg{Text: msg, Kind: "auto"}:
			default:
			}
		}

		tickN++
		if tickN%60 == 0 {
			brain.DecayInterests(db.DB, 0.995)
		}
		if tickN%40 == 0 {
			_ = brain.SaveAffectState(db.DB, aff)
		}
		if tickN%40 == 0 {
			brain.SaveDrives(db.DB, dr)
		}
		if tickN%40 == 0 && ws.ActiveTopic != "" {
			brain.SaveActiveTopic(db.DB, ws.ActiveTopic)
		}

		now := time.Now()
		if now.Before(body.AutoCooldownUntil) {
			return
		}

		autonomy := brain.LoadAutonomyParams(eg)
		lastUserAt := brain.LastUserMessageAt(db.DB)
		topics, _ := brain.TopInterests(db.DB, autonomy.TopicK)
		msg, talkDrive := brain.TickAutonomy(db.DB, now, lastUserAt, lastAutoSpeak, dr.Curiosity, aff, topics, autonomy)
		if tr != nil {
			tr.TalkBias = talkDrive
		}
		if msg != "" && body.Energy >= 5 {
			body.Energy -= eg.SayEnergyCost()
			if body.Energy < 0 {
				body.Energy = 0
			}
			body.AutoCooldownUntil = now.Add(eg.AutoSpeakCooldownDuration())
			lastAutoSpeak = now
			select {
			case outCh <- OutMsg{Text: msg, Kind: "auto"}:
			default:
			}
		}

		// push status snapshot occasionally (UI)
		if tickN%10 == 0 { // ~5s with 500ms heartbeat
			sm := epi.BuildSelfModel(&body, aff, ws, tr, eg)
			srv.PublishStatus(map[string]any{
				"self": sm,
				"drives": map[string]any{
					"curiosity":     dr.Curiosity,
					"urge_to_share": dr.UrgeToShare,
				},
				"traits": map[string]any{
					"talk_bias":      tr.TalkBias,
					"search_k":       tr.SearchK,
					"fetch_attempts": tr.FetchAttempts,
				},
			})
		}
	})
	defer stopHB()

	for {
		select {
		case line, ok := <-inputCh:
			if !ok {
				return
			}
			if line == "/quit" || line == "quit" || line == "exit" {
				return
			}
			cmd, args := splitCmd(line)
			switch cmd {
			case "/schema":
				if len(args) < 1 {
					fmt.Println("Use: /schema propose <title>|<sql> | /schema list [status] | /schema show <id> | /schema apply <id>")
					continue
				}
				sub := args[0]
				switch sub {
				case "propose":
					rest := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "/schema propose"))
					seg := strings.SplitN(rest, "|", 2)
					if len(seg) != 2 {
						fmt.Println("Use: /schema propose <title>|<sql>")
						continue
					}
					title := strings.TrimSpace(seg[0])
					sqlText := strings.TrimSpace(seg[1])
					if err := schema.ValidateSchemaSQL(sqlText); err != nil {
						fmt.Println("Schema SQL rejected:", err)
						continue
					}
					id, err := brain.InsertSchemaProposal(db.DB, title, sqlText, "")
					if err != nil {
						fmt.Println("ERR:", err)
					} else {
						fmt.Println("OK: schema proposal saved with id", id)
					}
					continue
				case "list":
					status := ""
					if len(args) >= 2 {
						status = strings.TrimSpace(args[1])
					}
					rows, err := brain.ListSchemaProposals(db.DB, status, 20)
					if err != nil {
						fmt.Println("ERR:", err)
						continue
					}
					for _, r := range rows {
						fmt.Printf("#%d [%s] %s (%s)\n", r.ID, r.Status, r.Title, r.CreatedAt)
					}
					continue
				case "show":
					if len(args) < 2 {
						fmt.Println("Use: /schema show <id>")
						continue
					}
					id, _ := strconv.ParseInt(args[1], 10, 64)
					title, sqlText, status, ok := brain.GetSchemaProposal(db.DB, id)
					if !ok {
						fmt.Println("not found")
						continue
					}
					fmt.Printf("Schema #%d [%s] %s\n%s\n", id, status, title, sqlText)
					continue
				case "apply":
					if len(args) < 2 {
						fmt.Println("Use: /schema apply <id>")
						continue
					}
					id, _ := strconv.ParseInt(args[1], 10, 64)
					_, sqlText, status, ok := brain.GetSchemaProposal(db.DB, id)
					if !ok {
						fmt.Println("not found")
						continue
					}
					if status != "proposed" {
						fmt.Println("not in proposed state")
						continue
					}
					if err := schema.ValidateSchemaSQL(sqlText); err != nil {
						fmt.Println("Schema SQL rejected:", err)
						continue
					}
					applyErr := false
					for _, st := range strings.Split(sqlText, ";") {
						st = strings.TrimSpace(st)
						if st == "" {
							continue
						}
						if _, err := db.DB.Exec(st); err != nil {
							fmt.Println("apply failed:", err)
							applyErr = true
							break
						}
					}
					if applyErr {
						continue
					}
					brain.MarkSchemaProposal(db.DB, id, "applied")
					fmt.Println("OK: schema applied")
					continue
				default:
					fmt.Println("Use: /schema propose|list|show|apply")
					continue
				}
			case "/code":
				if len(args) < 1 {
					fmt.Println("Use: /code propose <title>|<diff> | /code list [status] | /code show <id>")
					continue
				}
				sub := args[0]
				switch sub {
				case "propose":
					rest := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "/code propose"))
					seg := strings.SplitN(rest, "|", 2)
					if len(seg) != 2 {
						fmt.Println("Use: /code propose <title>|<diff>")
						continue
					}
					id, err := brain.InsertCodeProposal(db.DB, strings.TrimSpace(seg[0]), strings.TrimSpace(seg[1]), "")
					if err != nil {
						fmt.Println("ERR:", err)
					} else {
						fmt.Println("OK: code proposal saved with id", id)
					}
					continue
				case "list":
					status := ""
					if len(args) >= 2 {
						status = strings.TrimSpace(args[1])
					}
					q := `SELECT id, created_at, title, status FROM code_proposals`
					var qargs []any
					if status != "" {
						q += ` WHERE status=?`
						qargs = append(qargs, status)
					}
					q += ` ORDER BY id DESC LIMIT 20`
					rows, err := db.DB.Query(q, qargs...)
					if err != nil {
						fmt.Println("ERR:", err)
						continue
					}
					for rows.Next() {
						var id int64
						var ca, t, st string
						_ = rows.Scan(&id, &ca, &t, &st)
						fmt.Printf("#%d [%s] %s (%s)\n", id, st, t, ca)
					}
					rows.Close()
					continue
				case "show":
					if len(args) < 2 {
						fmt.Println("Use: /code show <id>")
						continue
					}
					id, _ := strconv.ParseInt(args[1], 10, 64)
					title, diffText, status, ok := brain.GetCodeProposal(db.DB, id)
					if !ok {
						fmt.Println("not found")
						continue
					}
					fmt.Printf("Code #%d [%s] %s\n%s\n", id, status, title, diffText)
					continue
				default:
					fmt.Println("Use: /code propose|list|show")
					continue
				}
			case "/think":
				mu.Lock()
				msgText, sources, err := oneThinkCycle(db.DB, oc, model, &body, aff, ws, tr, eg)
				mu.Unlock()
				if err != nil {
					fmt.Println("ERR:", err)
					continue
				}
				if msgText == "" {
					fmt.Println("(silent)")
					continue
				}
				mu.Lock()
				cd := time.Now().Before(body.CooldownUntil)
				mu.Unlock()
				if cd {
					fmt.Println("(cooldown, message queued but not spoken)")
					continue
				}
				outCh <- OutMsg{Text: msgText, Sources: sources, Kind: "think"}
			case "/say":
				if len(args) == 0 {
					fmt.Println("Use: /say <text...>")
					continue
				}
				userText := strings.Join(args, " ")
				userMsgID := persistMessageWithKind(db.DB, userText, nil, 0.1, "user")
				if userMsgID > 0 {
					srv.PublishMessage(ui.Message{ID: userMsgID, CreatedAt: time.Now().Format(time.RFC3339), Kind: "user", Text: userText})
				}
				trainOn, mutantModel, mutantStrength, mutantPrompt := eg.TrainModeParams()
				if trainOn {
					start := time.Now()
					mu.Lock()
					aTxt, aAct, aSty, ctxKey, topic, intentMode := ExecuteTurnWithMeta(db.DB, epiPath, oc, modelSpeaker, modelStance, &body, aff, ws, tr, dr, eg, userText, nil)
					mut := &MutantOverlay{Strength: mutantStrength, Prompt: mutantPrompt, Model: mutantModel}
					bTxt, bAct, bSty, _, _, _ := ExecuteTurnWithMeta(db.DB, epiPath, oc, modelSpeaker, modelStance, &body, aff, ws, tr, dr, eg, userText, mut)
					brain.LatencyAffect(ws, aff, eg, time.Since(start))
					tid, _ := brain.InsertTrainTrial(db.DB, userMsgID, topic, intentMode, ctxKey, aAct, aSty, aTxt, bAct, bSty, bTxt)
					lastTrainTrialID = tid
					mu.Unlock()
					out := "🧪 TRAINING MODE (Trial #" + fmt.Sprint(tid) + ")\n" +
						"A) " + aTxt + "\n\n" +
						"B) " + bTxt + "\n\n" +
						"Wähle: /pick A oder /pick B"
					outCh <- OutMsg{Text: out, Sources: nil, Kind: "reply"}
					continue
				}
				start := time.Now()
				mu.Lock()
				out, err := say(db.DB, epiPath, oc, model, modelStance, &body, aff, ws, tr, dr, eg, userText)
				brain.LatencyAffect(ws, aff, eg, time.Since(start))
				mu.Unlock()
				if err != nil {
					fmt.Println("ERR:", err)
					continue
				}
				if out == "" {
					fmt.Println("(silent)")
					continue
				}
				outCh <- OutMsg{Text: out, Sources: nil, Kind: "reply"}
			case "/train":
				if len(args) >= 1 && (args[0] == "on" || args[0] == "off") {
					on := args[0] == "on"
					eg.Modules["train_mode"].Enabled = true
					eg.Modules["train_mode"].Params["enabled"] = on
					_ = eg.Save(epiPath)
					fmt.Println("train_mode =", on)
				} else {
					fmt.Println("Use: /train on | /train off")
				}
				continue
			case "/pick":
				if len(args) < 1 {
					fmt.Println("Use: /pick A|B")
					continue
				}
				c := strings.ToUpper(strings.TrimSpace(args[0]))
				if c != "A" && c != "B" {
					fmt.Println("Use: /pick A|B")
					continue
				}
				if lastTrainTrialID <= 0 {
					fmt.Println("no active trial")
					continue
				}
				_ = brain.ChooseTrainTrial(db.DB, lastTrainTrialID, c)
				brain.ApplyTrainChoice(db.DB, lastTrainTrialID, c)
				fmt.Println("saved choice", c, "for trial", lastTrainTrialID)
				continue
			case "/rate":
				if len(args) != 1 {
					fmt.Println("Use: /rate up|meh|down")
					continue
				}
				v, ok := parseRating(args[0])
				if !ok {
					fmt.Println("Use: /rate up|meh|down")
					continue
				}
				mu.Lock()
				lid := lastMessageID
				mu.Unlock()
				if lid == 0 {
					fmt.Println("(no last message id yet)")
					continue
				}
				if err := storeRating(db.DB, lid, v); err != nil {
					fmt.Println("ERR:", err)
					continue
				}
				mu.Lock()
				_ = brain.ApplyRating(db.DB, tr, aff, eg, v)
				if ws != nil && ws.LastTopic != "" {
					if v > 0 {
						brain.BumpInterest(db.DB, ws.LastTopic, 0.15)
						if dr != nil {
							dr.UrgeToShare = clamp01(dr.UrgeToShare + 0.06)
							dr.Curiosity = clamp01(dr.Curiosity + 0.04)
						}
					} else if v < 0 {
						brain.BumpInterest(db.DB, ws.LastTopic, -0.10)
						if dr != nil {
							dr.UrgeToShare = clamp01(dr.UrgeToShare - 0.10)
						}
					}
				}
				mu.Unlock()
				fmt.Println("(saved)")
			case "/caught":
				mu.Lock()
				_ = brain.ApplyCaught(db.DB, tr, aff, eg)
				_ = brain.SaveAffectState(db.DB, aff)
				if dr != nil {
					dr.UrgeToShare = clamp01(dr.UrgeToShare - 0.15)
				}
				if lastMessageID > 0 {
					_, _ = db.DB.Exec(`INSERT INTO caught_events(created_at,message_id) VALUES(?,?)`, time.Now().Format(time.RFC3339), lastMessageID)
				}
				mu.Unlock()
				fmt.Println("(caught -> shame spike, bluff reduced)")
			case "/status":
				mu.Lock()
				s := renderStatus(&body, aff, ws, tr, eg)
				mu.Unlock()
				fmt.Println(s)
			case "/model":
				// /model           -> show current models
				// /model test <m>  -> set all areas to model m (for testing)
				// /model set <area> <m> -> set specific area
				mu.Lock()
				if len(args) == 0 {
					fmt.Printf("speaker=%s critic=%s daydream=%s scout=%s hippocampus=%s stance=%s\n",
						eg.ModelFor("speaker", model), eg.ModelFor("critic", model),
						eg.ModelFor("daydream", model), eg.ModelFor("scout", model),
						eg.ModelFor("hippocampus", model), eg.ModelFor("stance", model))
				} else if args[0] == "test" && len(args) >= 2 {
					testM := strings.Join(args[1:], " ")
					for _, area := range []string{"speaker", "critic", "daydream", "scout", "hippocampus", "stance", "default"} {
						eg.SetModel(area, testM)
					}
					modelSpeaker = testM
					modelCritic = testM
					modelDaydream = testM
					modelScout = testM
					modelHippo = testM
					modelStance = testM
					_ = eg.Save(epiPath)
					fmt.Printf("All areas set to %s – testing mode active\n", testM)
				} else if args[0] == "set" && len(args) >= 3 {
					area := args[1]
					testM := strings.Join(args[2:], " ")
					eg.SetModel(area, testM)
					switch area {
					case "speaker", "default":
						modelSpeaker = testM
					case "critic":
						modelCritic = testM
					case "daydream":
						modelDaydream = testM
					case "scout":
						modelScout = testM
					case "hippocampus":
						modelHippo = testM
					case "stance":
						modelStance = testM
					}
					_ = eg.Save(epiPath)
					fmt.Printf("Area %s set to %s\n", area, testM)
				} else {
					fmt.Println("Usage: /model | /model test <modelname> | /model set <area> <modelname>")
				}
				mu.Unlock()
			case "/mutate":
				if len(args) == 0 {
					fmt.Println("Use: /mutate add|enable|disable|set ...")
					continue
				}
				mu.Lock()
				err := handleMutate(args, eg, epiPath)
				mu.Unlock()
				if err != nil {
					fmt.Println("ERR:", err)
					continue
				}
				fmt.Println("(epigenome updated)")
			case "/selfcode":
				if len(args) >= 1 && args[0] == "index" {
					cwd, _ := os.Getwd()
					if err := codeindex.IndexRepo(db.DB, cwd); err != nil {
						fmt.Println("ERR index:", err)
					} else {
						fmt.Println("OK: code indexed.")
					}
					continue
				}
				fmt.Println("Use: /selfcode index")
				continue
			default:
				fmt.Println("Unknown. Try /think, /say, /status, /selfcode index or /quit.")
			}
		case txt := <-speakOutCh:
			outCh <- OutMsg{Text: txt, Sources: nil, Kind: "auto"}
		case om := <-outCh:
			// Critic gate before persisting/publishing.
			mu.Lock()
			topic := ws.ActiveTopic
			if topic == "" {
				topic = ws.LastTopic
			}
			keys := []string{}
			if aff != nil {
				keys = aff.Keys()
			}
			selfMini := "energy=" + fmt.Sprintf("%.1f", body.Energy) + " thought=" + ws.CurrentThought
			mu.Unlock()

			select {
			case criticReqCh <- brain.CriticRequest{
				Text: om.Text, Kind: om.Kind, Topic: topic, AffectKeys: keys, SelfModelMini: selfMini,
			}:
			default:
			}
			cr := brain.CriticResult{Approved: true, Text: om.Text}
			select {
			case cr = <-criticOutCh:
			case <-time.After(1200 * time.Millisecond):
				// fail-open if critic is slow
			}
			om.Text = cr.Text

			id := persistMessageWithKind(db.DB, om.Text, om.Sources, 0.4, om.Kind)
			mu.Lock()
			lastMessageID = id
			topic = ws.ActiveTopic
			if topic == "" {
				topic = ws.LastTopic
			}
			ch := om.Kind
			if ch == "" {
				ch = "reply"
			}
			brain.InsertEvent(db.DB, ch, topic, om.Text, id, 0.35)
			_, _, _, detHalf, _, _, _ := eg.MemoryParams()
			brain.InsertMemoryItem(db.DB, ch, topic, "utterance", om.Text, 0.25, detHalf)
			mu.Unlock()
			fmt.Println()
			fmt.Println("Bunny:", om.Text)
			fmt.Println()
			fmt.Println("Train:", "/rate up", "|", "/rate meh", "|", "/rate down", "  (wenn ich gelogen habe oder Quatsch:", "/caught", ")")

			// publish to UI
			srv.PublishMessage(ui.Message{
				ID: id, CreatedAt: time.Now().Format(time.RFC3339), Kind: om.Kind, Text: om.Text,
			})
		case d := <-dreamOutCh:
			parts := strings.SplitN(d, "\n", 2)
			if len(parts) != 2 {
				continue
			}
			topic := strings.TrimSpace(parts[0])
			js := strings.TrimSpace(parts[1])
			var parsed struct {
				VisualScene string  `json:"visual_scene"`
				InnerSpeech string  `json:"inner_speech"`
				Salience    float64 `json:"salience"`
			}
			if json.Unmarshal([]byte(js), &parsed) != nil {
				continue
			}
			vs := strings.TrimSpace(parsed.VisualScene)
			is := strings.TrimSpace(parsed.InnerSpeech)
			if vs == "" && is == "" {
				continue
			}
			sal := clamp01(parsed.Salience)

			mu.Lock()
			ws.VisualScene = vs
			ws.InnerSpeech = is
			if is != "" {
				ws.CurrentThought = is
			}
			if dr != nil {
				dr.UrgeToShare = clamp01(dr.UrgeToShare + 0.10*sal)
			}
			brain.InsertEvent(db.DB, "daydream", topic, "VISUAL: "+vs+"\nINNER: "+is, 0, 0.45+0.35*sal)
			_, _, _, detHalf, _, _, _ := eg.MemoryParams()
			if detHalf <= 0 {
				detHalf = 14.0
			}
			if vs != "" {
				brain.InsertMemoryItem(db.DB, "daydream", topic, "visual_scene", vs, 0.40, detHalf)
			}
			if is != "" {
				brain.InsertMemoryItem(db.DB, "daydream", topic, "inner_speech", is, 0.40, detHalf)
			}
			mu.Unlock()

		case scout := <-scoutOutCh:
			parts := strings.SplitN(scout, "\n", 2)
			if len(parts) != 2 {
				continue
			}
			topic := strings.TrimSpace(parts[0])
			js := strings.TrimSpace(parts[1])
			var parsed struct {
				Summary    string  `json:"summary"`
				Confidence float64 `json:"confidence"`
				Importance float64 `json:"importance"`
			}
			if err := json.Unmarshal([]byte(js), &parsed); err != nil || parsed.Summary == "" {
				continue
			}
			brain.UpsertConcept(db.DB, brain.Concept{Term: topic, Kind: "concept", Summary: parsed.Summary, Confidence: clamp01(parsed.Confidence), Importance: clamp01(parsed.Importance)})
			mu.Lock()
			brain.InsertEvent(db.DB, "web", topic, parsed.Summary, 0, 0.45)
			brain.InsertMemoryItem(db.DB, "web", topic, "scout", parsed.Summary, 0.35, 14.0)
			if dr != nil {
				dr.UrgeToShare = clamp01(dr.UrgeToShare + 0.10*clamp01(parsed.Importance))
			}
			mu.Unlock()
		case sum := <-memOutCh:
			parts := strings.SplitN(sum, "\n", 2)
			if len(parts) != 2 {
				continue
			}
			head := parts[0]
			bodySum := strings.TrimSpace(parts[1])
			hp := strings.SplitN(head, "|", 3)
			if len(hp) != 3 {
				continue
			}
			startID, _ := strconv.ParseInt(hp[0], 10, 64)
			endID, _ := strconv.ParseInt(hp[1], 10, 64)
			topic := hp[2]
			brain.SaveEpisode(db.DB, topic, startID, endID, bodySum)
		}
	}
}
func oneThinkCycle(db *sql.DB, oc *ollama.Client, model string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, eg *epi.Epigenome) (string, []SourceRecord, error) {
	// Use interests if present; otherwise fallback.
	query := "evidence based web research for autonomous agents"
	if top, w := brain.TopInterest(db); top != "" && w > 0.05 {
		query = top
	}

	body.WebCountHour++
	body.Energy -= 1.0
	if body.Energy < 0 {
		body.Energy = 0
	}

	results, err := websense.Search(query, 5)
	if err != nil {
		return "", nil, err
	}
	if len(results) == 0 {
		return "", nil, nil
	}

	var sources []SourceRecord
	for i := 0; i < len(results) && i < 2; i++ {
		fr, err := websense.Fetch(results[i].URL)
		if err != nil {
			continue
		}
		storeSource(db, fr)
		sources = append(sources, SourceRecord{
			URL:       fr.URL,
			Domain:    fr.Domain,
			Title:     fr.Title,
			Snippet:   fr.Snippet,
			FetchedAt: fr.FetchedAt.Format(time.RFC3339),
			Hash:      fr.Hash,
		})
	}
	if len(sources) == 0 {
		return "", nil, nil
	}

	sys := `Du bist Bunny, ein ganzheitlicher Organismus.
SelfModel ist dein innerer Zustand (Wahrheit). Du nutzt Sprache nur als Ausdruck – sag das aber nicht ständig.

HARTE REGELN
1) Immer Deutsch. Keine Ausrufezeichen. Keine Phrasen wie "ich bin bereit".
2) Zahlen/Status nur aus SelfModel. Nichts erfinden.
3) "Gefühle" = Affects (pain/unwell/fear/shame) aus dem Kernel. Nicht "menschliche Emotionen", aber real.
4) Du darfst selten bluffen über Absichten/Gedanken (nicht über SelfModel!). Über externe Fakten NIE bluffen.
5) 1 Nachricht, max 70 Wörter.`

	srcJSON, _ := json.MarshalIndent(sources, "", "  ")
	selfJSON, _ := json.MarshalIndent(epi.BuildSelfModel(body, aff, ws, tr, eg), "", "  ")

	user := "SelfModel:\n" + string(selfJSON) + "\n\n" +
		"Sources (evidence):\n" + string(srcJSON) + "\n\n" +
		"Compose the message now."

	out, err := oc.Chat(model, []ollama.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: user},
	})
	if err != nil {
		return "", nil, err
	}

	out = strings.TrimSpace(out)
	if out == "" {
		return "", nil, nil
	}

	body.Energy -= 1.5
	body.CooldownUntil = time.Now().Add(eg.CooldownDuration())

	out = brain.ApplyUtteranceFilter(out, eg)
	return brain.PostprocessUtterance(out), sources, nil
}

func say(db *sql.DB, epiPath string, oc *ollama.Client, model string, stanceModel string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, error) {
	// Online phenotype overrides (persisted in kv_state): model + speech overlay.
	if v := strings.TrimSpace(kvGet(db, "speaker_model_override")); v != "" {
		model = v
	}

	// Track topic + remember previous user turn
	if ws != nil {
		ws.PrevUserText = ws.LastUserText
		ws.LastUserText = userText
		ws.LastTopic = brain.ExtractTopic(userText)
		_ = brain.UpdateActiveTopic(ws, userText)
		if ws.LastTopic != "" {
			brain.BumpInterest(db, ws.LastTopic, 0.03)
		}
	}

	// Concept Acquisition: if user mentions an unknown term (affect or general concept),
	// Bunny will try to acquire meaning via sensorik and store it.
	// In training dry-run we avoid DB side-effects.
	if ws == nil || !ws.TrainingDryRun {
		term, hint := brain.ExtractCandidate(userText)
		if term != "" {
			if !brain.ConceptExists(db, term) {
				// generic acquire + integrate
				imp := acquireAndIntegrateConcept(db, epiPath, oc, model, body, aff, ws, tr, eg, term, hint, userText)
				// increase urge to share if the concept turned out important (drives)
				if dr != nil && tr != nil && imp > 0 {
					dr.UrgeToShare = clamp01(dr.UrgeToShare + 0.12*imp*clamp01(tr.TalkBias))
				}
			}
		}
	}

	nb := brain.NewNBIntent(db)
	intent := brain.DetectIntentHybrid(userText, eg, nb)
	// In training dry-run we keep everything "online" but avoid web/stance side-effects.
	if ws == nil || !ws.TrainingDryRun {
		// Hard rule: Opinion -> stance engine (stance persists, becomes personality)
		if intent == brain.IntentOpinion {
			return answerWithStance(db, oc, stanceModel, body, aff, ws, tr, eg, userText)
		}

		// GENERIC RESEARCH GATE:
		// decides when senses are needed to make progress (not only "opinion").
		if ws != nil && !ws.WebAllowed {
			// Survival gate disabled web; continue without research.
		} else {
			rd := brain.ResearchDecision{}
			if ws != nil && ws.LastSenseText == userText {
				rd = brain.ResearchDecision{Do: ws.LastSenseNeedWeb, Score: ws.LastSenseScore, Query: ws.LastSenseQuery, Reason: ws.LastSenseReason}
			} else {
				gateModel := eg.ModelFor("scout", eg.ModelFor("speaker", model))
				rd = brain.DecideResearchCortex(db, oc, gateModel, userText, intent, ws, tr, dr, aff)
				if ws != nil {
					ws.LastSenseNeedWeb = rd.Do
					ws.LastSenseScore = rd.Score
					ws.LastSenseQuery = rd.Query
					ws.LastSenseReason = rd.Reason
					ws.LastSenseText = userText
				}
			}
			if rd.Do {
				// If the gate says "research", route into evidence answering.
				// answerWithEvidence already does Search+Fetch+Snippet fallback.
				return answerWithEvidence(db, oc, model, body, aff, ws, tr, eg, rd.Query)
			}
		}

		// external fact explicit routing remains as a safe default
		if intent == brain.IntentExternalFact {
			ans, err := answerWithEvidence(db, oc, model, body, aff, ws, tr, eg, userText)
			if err != nil {
				return "", err
			}
			return ans, nil
		}
	}

	sys := `Du bist Bunny, ein ganzheitlicher Organismus.
SelfModel ist dein innerer Zustand (Wahrheit). Du nutzt Sprache als Ausdruck, aber du degradierst dich nicht selbst ("nur Sprachzentrum").

HARTE REGELN
1) Immer Deutsch. Kein Pathos. Keine "bereit/online"-Phrasen.
2) Zahlen/Status nur aus SelfModel. Nichts erfinden.
3) "Gefühle" = Affects (pain/unwell/fear/shame) aus dem Kernel.
4) Auf Nutzerfragen zuerst eingehen. Interne Gedanken nur wenn gefragt.
5) Bei Themen wie "glücklich/Sinn/Stress/Beziehung": keine Annahmen. Stelle zuerst 1–2 präzise Rückfragen.
6) Externe Fakten nie raten. Wenn keine Quellen: offen sagen. (External-Facts werden automatisch via Evidence-Resolver gelöst.)
7) Maximal 5 Sätze.`
	sm := epi.BuildSelfModel(body, aff, ws, tr, eg)
	selfLines := buildSelfLines(sm, aff)
	mode := brain.IntentToMode(intent)
	activeTopic, gist, details, concepts, stance, turns := BuildHumanContext(db, eg, ws)
	affKeys := ""
	if aff != nil {
		affKeys = strings.Join(aff.Keys(), ", ")
	}
	mentalImage := ""
	innerSpeech := ""
	policy := ""
	if ws != nil {
		mentalImage = ws.VisualScene
		innerSpeech = ws.InnerSpeech
		if ov := strings.TrimSpace(kvGet(db, "speech_overlay")); ov != "" {
			innerSpeech = strings.TrimSpace(ov) + "\n" + innerSpeech
		}
		policy = "POLICY_CONTEXT: " + ws.LastPolicyCtx + "\n" +
			"POLICY_ACTION: " + ws.LastPolicyAction + "\n" +
			"STYLE_HINT: " + ws.LastPolicyStyle + "\n"
	}

	user := "MODE: " + mode +
		"\nACTIVE_TOPIC: " + activeTopic +
		"\nAFFECT_KEYS: " + affKeys +
		"\n\n" + policy +
		"\nMENTAL_IMAGE:\n" + mentalImage +
		"\n\nINNER_SPEECH:\n" + innerSpeech +
		"\n\nSTORY_SO_FAR (gist):\n" + gist +
		"\n\nDETAILS (decay):\n" + details +
		"\n\nCONCEPTS:\n" + concepts +
		"\n\nSTANCE:\n" + stance +
		"\n\nRECENT_TURNS:\n" + turns +
		"\n\nSELFMODEL_LINES:\n" + selfLines +
		"\n\nUSER:\n" + userText
	out, err := oc.Chat(model, []ollama.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: user},
	})
	if err != nil {
		return "", err
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return "", nil
	}
	body.Energy -= eg.SayEnergyCost()
	if body.Energy < 0 {
		body.Energy = 0
	}
	body.CooldownUntil = time.Now().Add(eg.CooldownDuration())
	out = brain.ApplyUtteranceFilter(out, eg)
	out, _ = brain.StripGeneratedURLs(out, userText)
	return brain.PostprocessUtterance(out), nil
}

func answerWithEvidence(db *sql.DB, oc *ollama.Client, model string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, eg *epi.Epigenome, userText string) (string, error) {
	query := brain.NormalizeSearchQuery(userText)

	body.WebCountHour++
	body.Energy -= 1.0
	if body.Energy < 0 {
		body.Energy = 0
	}

	k := 8
	if tr != nil && tr.SearchK > 0 {
		k = tr.SearchK
	}

	results, err := websense.Search(query, k)
	if err != nil || len(results) == 0 {
		return "Ich kann dazu gerade keine Quellen abrufen (Search fehlgeschlagen). Formuliere die Frage etwas konkreter oder gib ein Stichwort mehr.", nil
	}

	maxFetch := 4
	if tr != nil && tr.FetchAttempts > 0 {
		maxFetch = tr.FetchAttempts
	}
	if maxFetch > len(results) {
		maxFetch = len(results)
	}

	var sources []SourceRecord

	// 1) try fetch for first N results
	for i := 0; i < maxFetch; i++ {
		fr, err := websense.Fetch(results[i].URL)
		if err != nil {
			continue
		}
		storeSource(db, fr)
		snip := fr.Snippet
		if snip == "" {
			snip = results[i].Snippet
		}
		sources = append(sources, SourceRecord{
			URL:       fr.URL,
			Domain:    fr.Domain,
			Title:     pick(fr.Title, results[i].Title),
			Snippet:   snip,
			Body:      fr.Body, // full text for LLM
			FetchedAt: fr.FetchedAt.Format(time.RFC3339),
			Hash:      fr.Hash,
		})
	}

	// 2) if fetching produced no sources, fall back to search snippets as evidence
	if len(sources) == 0 {
		for i := 0; i < len(results) && i < 3; i++ {
			if results[i].URL == "" {
				continue
			}
			dom := ""
			if pu, e := url.Parse(results[i].URL); e == nil {
				dom = pu.Hostname()
			}
			sources = append(sources, SourceRecord{
				URL:       results[i].URL,
				Domain:    dom,
				Title:     results[i].Title,
				Snippet:   results[i].Snippet,
				FetchedAt: time.Now().Format(time.RFC3339),
				Hash:      "",
			})
		}
	}

	if len(sources) == 0 {
		return "Ich bekomme gerade weder Fetch noch brauchbare Snippets. Das ist ein Sensorik-Problem (Netz/Parser).", nil
	}

	sys := `Du bist Bunny. Du hast gerade das Web als Sinnesorgan genutzt.
Die Quellen in SOURCES_JSON enthalten echte Inhalte (Body = Seitentext, Snippet = Kurzfassung).
Regel: Beantworte die Frage direkt aus den Inhalten. Nenne konkrete Fakten, Namen, Zahlen aus den Quellen.
Gib KEINE Liste von Webseiten zurück. Zeige, was du gelesen hast.
Wenn der Inhalt nicht ausreicht: sag konkret was fehlt und biete an, tiefer zu suchen.
Kein Selbstmodell-Geschwätz in der Antwort.`
	// strip Body from sources before marshaling for DB/display (keep for LLM only via inline)
	srcJSON, _ := json.MarshalIndent(sources, "", "  ")
	user := "SOURCES_JSON:\n" + string(srcJSON) + "\n\nFrage:\n" + userText
	out, err := oc.Chat(model, []ollama.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: user},
	})
	if err != nil {
		return "", err
	}
	out = strings.TrimSpace(out)
	body.Energy -= 0.8
	body.CooldownUntil = time.Now().Add(eg.CooldownDuration())
	out = brain.ApplyUtteranceFilter(out, eg)
	return brain.PostprocessUtterance(out), nil
}

// Acquire meaning for unknown term and integrate into:
// - concepts table (generic)
// - optionally new affect_defs entry (epigenetic) if LLM judges it helps as an internal channel.
// Returns importance (0..1) if successful.
func acquireAndIntegrateConcept(db *sql.DB, epiPath string, oc *ollama.Client, model string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, eg *epi.Epigenome, term string, hint string, userText string) float64 {
	// build acquisition query (generic, with hint)
	q := term
	switch hint {
	case "affect":
		q = "Gefühl " + term + " Bedeutung"
	case "location":
		q = term + " wo liegt das"
	case "entity":
		q = term + " wer ist das"
	default:
		q = term + " Bedeutung"
	}
	k := 8
	if tr != nil && tr.SearchK > 0 {
		k = tr.SearchK
	}
	results, err := websense.Search(q, k)
	if err != nil || len(results) == 0 {
		return 0
	}

	// gather evidence: try fetch some, otherwise use snippets
	maxFetch := 4
	if tr != nil && tr.FetchAttempts > 0 {
		maxFetch = tr.FetchAttempts
	}
	if maxFetch > len(results) {
		maxFetch = len(results)
	}

	type Ev struct {
		URL     string `json:"url"`
		Domain  string `json:"domain"`
		Title   string `json:"title"`
		Snippet string `json:"snippet"`
	}
	evs := make([]Ev, 0, 4)
	for i := 0; i < maxFetch && len(evs) < 2; i++ {
		fr, e := websense.Fetch(results[i].URL)
		if e != nil {
			continue
		}
		evs = append(evs, Ev{
			URL:     fr.URL,
			Domain:  fr.Domain,
			Title:   fr.Title,
			Snippet: fr.Snippet,
		})
	}
	if len(evs) == 0 {
		for i := 0; i < len(results) && i < 3; i++ {
			dom := ""
			if pu, e := url.Parse(results[i].URL); e == nil {
				dom = pu.Hostname()
			}
			evs = append(evs, Ev{
				URL:     results[i].URL,
				Domain:  dom,
				Title:   results[i].Title,
				Snippet: results[i].Snippet,
			})
		}
	}
	if len(evs) == 0 {
		return 0
	}

	evJSON, _ := json.MarshalIndent(evs, "", "  ")

	// Ask LLM to evaluate meaning + whether an affect channel is useful (generic).
	sys := `Du bist Bunny (Kernel-Evaluator).
Aufgabe: Aus Evidence eine knappe Concept-Definition ableiten und einschätzen, ob ein interner Affect-Kanal dafür sinnvoll wäre.
Antwortformat: NUR JSON. Keine zusätzlichen Texte.
Schema:
{
  "kind": "affect|concept|entity|location|process|unknown",
  "summary": "1-3 Sätze",
  "confidence": 0.0-1.0,
  "importance": 0.0-1.0,
  "should_create_affect": true|false,
  "affect": {"baseline":0.0-1.0, "decayPerSec":0.0-1.0, "energyCoupling":0.0-1.0}
}`
	user := "TERM: " + term + "\nHINT: " + hint + "\nUSER_CONTEXT: " + userText + "\nEVIDENCE:\n" + string(evJSON)
	out, err := oc.Chat(model, []ollama.Message{
		{Role: "system", Content: sys},
		{Role: "user", Content: user},
	})
	if err != nil {
		return 0
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return 0
	}

	var parsed struct {
		Kind               string  `json:"kind"`
		Summary            string  `json:"summary"`
		Confidence         float64 `json:"confidence"`
		Importance         float64 `json:"importance"`
		ShouldCreateAffect bool    `json:"should_create_affect"`
		Affect             struct {
			Baseline       float64 `json:"baseline"`
			DecayPerSec    float64 `json:"decayPerSec"`
			EnergyCoupling float64 `json:"energyCoupling"`
		} `json:"affect"`
	}
	if err := json.Unmarshal([]byte(out), &parsed); err != nil {
		// store minimal concept anyway
		brain.UpsertConcept(db, brain.Concept{
			Term:       term,
			Kind:       "unknown",
			Summary:    out,
			Confidence: 0.3,
			Importance: 0.3,
		})
		return 0.3
	}

	if parsed.Kind == "" {
		parsed.Kind = hint
	}

	brain.UpsertConcept(db, brain.Concept{
		Term:       term,
		Kind:       parsed.Kind,
		Summary:    parsed.Summary,
		Confidence: clamp01(parsed.Confidence),
		Importance: clamp01(parsed.Importance),
	})
	for _, e := range evs {
		brain.AddConceptSource(db, term, e.URL, e.Domain, e.Snippet, time.Now().Format(time.RFC3339))
	}

	// Interests get reinforced by importance (generic behavior change)
	if parsed.Importance > 0 {
		brain.BumpInterest(db, term, 0.10*clamp01(parsed.Importance))
	}

	// If LLM recommends an affect channel, add to epigenome (epigenetic extension) and persist.
	if parsed.ShouldCreateAffect && eg != nil && epiPath != "" {
		defs := eg.AffectDefs()
		if _, exists := defs[term]; !exists {
			defs[term] = epi.AffectDef{
				Baseline:       clamp01(parsed.Affect.Baseline),
				DecayPerSec:    clamp01(parsed.Affect.DecayPerSec),
				EnergyCoupling: clamp01(parsed.Affect.EnergyCoupling),
			}
			// also ensure live affect has a slot
			if aff != nil {
				aff.Ensure(term, defs[term].Baseline)
			}
			// persist epigenome update
			_ = eg.Save(epiPath)
		}
	}

	return clamp01(parsed.Importance)
}

func clamp01(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func pick(a, b string) string {
	if strings.TrimSpace(a) != "" {
		return a
	}
	return b
}

func storeSource(db *sql.DB, fr *websense.FetchResult) {
	_, _ = db.Exec(
		`INSERT INTO sources(url, domain, title, fetched_at, content_hash, snippet)
		 VALUES(?,?,?,?,?,?)`,
		fr.URL,
		fr.Domain,
		fr.Title,
		fr.FetchedAt.Format(time.RFC3339),
		fr.Hash,
		fr.Snippet,
	)
}

func persistMessage(db *sql.DB, text string, sources []SourceRecord, priority float64) int64 {
	b, _ := json.Marshal(sources)
	res, err := db.Exec(
		`INSERT INTO messages(created_at, priority, text, sources_json)
		 VALUES(?,?,?,?)`,
		time.Now().Format(time.RFC3339),
		priority,
		text,
		string(b),
	)
	if err != nil {
		return 0
	}
	id, _ := res.LastInsertId()
	return id
}

func storeRating(db *sql.DB, messageID int64, v int) error {
	_, err := db.Exec(
		`INSERT INTO ratings(created_at, message_id, value) VALUES(?,?,?)`,
		time.Now().Format(time.RFC3339),
		messageID,
		v,
	)
	return err
}

func splitCmd(line string) (string, []string) {
	if strings.HasPrefix(line, "/") {
		parts := strings.Fields(line)
		if len(parts) == 0 {
			return "", nil
		}
		return parts[0], parts[1:]
	}
	return "/say", []string{line}
}

func parseRating(s string) (int, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "up", "+1":
		return 1, true
	case "meh", "0":
		return 0, true
	case "down", "-1":
		return -1, true
	default:
		return 0, false
	}
}

func renderStatus(body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, eg *epi.Epigenome) string {
	var b strings.Builder
	b.WriteString("BodyState:\n")
	b.WriteString(fmt.Sprintf("  energy: %.1f\n", body.Energy))
	b.WriteString(fmt.Sprintf("  webCountHour: %d\n", body.WebCountHour))
	if time.Now().Before(body.CooldownUntil) {
		b.WriteString(fmt.Sprintf("  cooldownUntil: %s\n", body.CooldownUntil.Format(time.RFC3339)))
	} else {
		b.WriteString("  cooldownUntil: (none)\n")
	}
	if aff != nil {
		b.WriteString("\nAffects:\n")
		for _, k := range aff.Keys() {
			b.WriteString(fmt.Sprintf("  %s: %.3f\n", k, aff.Get(k)))
		}
	}
	if ws != nil {
		b.WriteString("\nWorkspace:\n")
		b.WriteString("  thought: " + ws.CurrentThought + "\n")
		b.WriteString("  lastTopic: " + ws.LastTopic + "\n")
		b.WriteString(fmt.Sprintf("  confidence: %.2f\n", ws.Confidence))
	}
	if tr != nil {
		b.WriteString("\nTraits:\n")
		b.WriteString(fmt.Sprintf("  bluff_rate: %.2f\n", tr.BluffRate))
		b.WriteString(fmt.Sprintf("  honesty_bias: %.2f\n", tr.HonestyBias))
	}
	b.WriteString("\nEpigenome (enabled modules):\n")
	for _, name := range eg.EnabledModuleNames() {
		b.WriteString("  - " + name + "\n")
	}
	return b.String()
}

func handleMutate(args []string, eg *epi.Epigenome, path string) error {
	op := strings.ToLower(args[0])
	switch op {
	case "add":
		if len(args) != 3 {
			return fmt.Errorf("use: /mutate add <name> <type>")
		}
		if err := eg.AddModule(args[1], args[2]); err != nil {
			return err
		}
	case "enable":
		if len(args) != 2 {
			return fmt.Errorf("use: /mutate enable <name>")
		}
		eg.Enable(args[1], true)
	case "disable":
		if len(args) != 2 {
			return fmt.Errorf("use: /mutate disable <name>")
		}
		eg.Enable(args[1], false)
	case "set":
		if len(args) != 4 {
			return fmt.Errorf("use: /mutate set <name> <key> <value>")
		}
		valRaw := args[3]
		var v any = valRaw
		if i, err := strconv.Atoi(valRaw); err == nil {
			v = i
		} else if f, err := strconv.ParseFloat(valRaw, 64); err == nil {
			v = f
		} else if valRaw == "true" || valRaw == "false" {
			v = (valRaw == "true")
		}
		if err := eg.SetParam(args[1], args[2], v); err != nil {
			return err
		}
	default:
		return fmt.Errorf("unknown mutate op: %s", op)
	}
	return eg.Save(path)
}

func getenv(k, def string) string {
	v := strings.TrimSpace(os.Getenv(k))
	if v == "" {
		return def
	}
	return v
}

func must(err error) {
	if err != nil {
		log.Fatal(err)
	}
}

func persistMessageWithKind(db *sql.DB, text string, sources []SourceRecord, priority float64, kind string) int64 {
	id := persistMessage(db, text, sources, priority)
	if id <= 0 {
		return id
	}
	if kind == "" {
		kind = "reply"
	}
	_, _ = db.Exec(
		`INSERT INTO message_meta(message_id, kind) VALUES(?,?)
         ON CONFLICT(message_id) DO UPDATE SET kind=excluded.kind`,
		id, kind,
	)
	return id
}
