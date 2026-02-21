package main

import (
	"bufio"
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
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
	"frankenstein-v0/internal/state"
	"frankenstein-v0/internal/websense"
)

type BodyState struct {
	Energy        float64 // 0..100
	MemLoad       float64 // proxy
	WebCountHour  int
	CooldownUntil time.Time
}

type SourceRecord struct {
	URL       string `json:"url"`
	Domain    string `json:"domain"`
	Title     string `json:"title"`
	Snippet   string `json:"snippet"`
	FetchedAt string `json:"fetched_at"`
	Hash      string `json:"hash"`
}

func main() {
	model := getenv("FRANK_MODEL", "llama3.1:8b")
	ollamaURL := getenv("OLLAMA_URL", "http://localhost:11434")
	dbPath := getenv("FRANK_DB", "data/frankenstein.sqlite")
	epiPath := getenv("FRANK_EPI", "data/epigenome.json")

	_ = os.MkdirAll("data", 0o755)

	db, err := state.Open(dbPath)
	must(err)
	defer db.Close()

	oc := ollama.New(ollamaURL)

	eg, err := epi.LoadOrInit(epiPath)
	if err != nil {
		log.Fatal(err)
	}

	// v0 BodyState
	body := BodyState{
		Energy:        80,
		MemLoad:       0,
		WebCountHour:  0,
		CooldownUntil: time.Time{},
	}

	// Kernel affect state (generic: pain/unwell/fear/...)
	aff := brain.NewAffectState()
	ws := brain.NewWorkspace()

	// traits (persistent learning knobs)
	tr, err := brain.LoadOrInitTraits(db.DB)
	must(err)

	// drives (curiosity, urge_to_share)
	dr, err := brain.LoadOrInitDrives(db.DB)
	must(err)

	// Load persisted affects
	_ = brain.LoadAffectState(db.DB, aff)

	// Shared lock because heartbeat and CLI touch body/aff/ws/tr
	var mu sync.Mutex

	fmt.Println("Bunny v0 online.")
	fmt.Println("Commands: /think | /say <text...> | /rate <up|meh|down> | /caught | /status | /mutate ... | /quit")
	fmt.Println()

	var lastMessageID int64 = 0
	reader := bufio.NewReader(os.Stdin)

	// Heartbeat: kernel ticks regardless of LLM usage
	hb := brain.NewHeartbeat(eg)
	var tickN int
	stopHB := hb.Start(func(delta time.Duration) {
		mu.Lock()
		defer mu.Unlock()

		// 1) Update affect loop (homeostasis)
		brain.TickAffects(&body, aff, eg, delta)

		// 2) Lightweight energy drift / recovery (epigenetic)
		brain.TickBody(&body, eg, delta)
		brain.TickWorkspace(ws, &body, aff, tr, eg, delta)
		brain.TickDrives(dr, aff, delta)
		brain.TickDaydream(db.DB, ws, dr, aff, delta)

		// interest decay (slow) + persist affects periodically
		tickN++
		if tickN%60 == 0 { // ~ every 30s if heartbeat=500ms
			brain.DecayInterests(db.DB, 0.995)
		}
		if tickN%40 == 0 { // persist affects ~ every 20s
			_ = brain.SaveAffectState(db.DB, aff)
		}
		if tickN%40 == 0 {
			brain.SaveDrives(db.DB, dr)
		}
	})
	defer stopHB()

	for {
		fmt.Print("> ")
		line, _ := reader.ReadString('\n')
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if line == "/quit" || line == "quit" || line == "exit" {
			return
		}

		cmd, args := splitCmd(line)

		switch cmd {
		case "/think":
			mu.Lock()
			// one “idle” cycle: pick a topic, search, fetch, propose a message
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
			lastMessageID = persistMessage(db.DB, msgText, sources, 0.5)
			fmt.Println()
			fmt.Println("Bunny:", msgText)
			fmt.Println()
			fmt.Println("Rate it with: /rate up | /rate meh | /rate down")
		case "/say":
			if len(args) == 0 {
				fmt.Println("Use: /say <text...>")
				continue
			}
			userText := strings.Join(args, " ")
			mu.Lock()
			out, err := say(db.DB, epiPath, oc, model, &body, aff, ws, tr, dr, eg, userText)
			mu.Unlock()
			if err != nil {
				fmt.Println("ERR:", err)
				continue
			}
			if out == "" {
				fmt.Println("(silent)")
				continue
			}
			lastMessageID = persistMessage(db.DB, out, nil, 0.2)
			fmt.Println()
			fmt.Println("Bunny:", out)
			fmt.Println()
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
			if lastMessageID == 0 {
				fmt.Println("(no last message id yet)")
				continue
			}
			if err := storeRating(db.DB, lastMessageID, v); err != nil {
				fmt.Println("ERR:", err)
				continue
			}
			mu.Lock()
			_ = brain.ApplyRating(db.DB, tr, aff, eg, v)
			// ratings also move interests: reinforce last topic if available
			if ws != nil && ws.LastTopic != "" {
				if v > 0 {
					brain.BumpInterest(db.DB, ws.LastTopic, 0.15)
				} else if v < 0 {
					brain.BumpInterest(db.DB, ws.LastTopic, -0.10)
				}
			}
			mu.Unlock()
			fmt.Println("(saved)")
		case "/caught":
			mu.Lock()
			_ = brain.ApplyCaught(db.DB, tr, aff, eg)
			_ = brain.SaveAffectState(db.DB, aff)
			mu.Unlock()
			fmt.Println("(caught -> shame spike, bluff reduced)")
		case "/status":
			mu.Lock()
			s := renderStatus(&body, aff, ws, tr, eg)
			mu.Unlock()
			fmt.Println(s)
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
		default:
			fmt.Println("Unknown. Try /think, /say, /status or /quit.")
		}

		_ = lastMessageID
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

func say(db *sql.DB, epiPath string, oc *ollama.Client, model string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, error) {
	// Track topic for workspace thinking (kernel-side)
	if ws != nil {
		ws.LastTopic = brain.ExtractTopic(userText)
		// baseline interest bump on any user topic
		if ws.LastTopic != "" {
			brain.BumpInterest(db, ws.LastTopic, 0.03)
		}
	}

	// Concept Acquisition: if user mentions an unknown term (affect or general concept),
	// Bunny will try to acquire meaning via sensorik and store it.
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

	intent := brain.DetectIntent(userText)
	if intent == brain.IntentExternalFact {
		ans, err := answerWithEvidence(db, oc, model, body, aff, ws, tr, eg, userText)
		if err != nil {
			return "", err
		}
		return ans, nil
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
	selfJSON, _ := json.MarshalIndent(epi.BuildSelfModel(body, aff, ws, tr, eg), "", "  ")
	mode := brain.IntentToMode(intent)
	user := "MODE: " + mode + "\n\nSelfModel:\n" + string(selfJSON) + "\n\nOliver says:\n" + userText
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

	sys := `Du bist Bunny. Nutze deine Sinnesorgane.
Regel: Externe Fakten nur aus SOURCES_JSON ableiten. Wenn SOURCES_JSON nicht reicht: eine Rückfrage oder offen sagen, was fehlt.`
	selfJSON, _ := json.MarshalIndent(epi.BuildSelfModel(body, aff, ws, tr, eg), "", "  ")
	srcJSON, _ := json.MarshalIndent(sources, "", "  ")
	user := "SelfModel:\n" + string(selfJSON) + "\n\nSOURCES_JSON:\n" + string(srcJSON) + "\n\nFrage:\n" + userText
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
