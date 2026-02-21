package main

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
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

	// Shared lock because heartbeat and CLI touch body/aff
	var mu sync.Mutex

	fmt.Println("Bunny v0 online.")
	fmt.Println("Commands: /think | /say <text...> | /rate <up|meh|down> | /caught | /status | /mutate ... | /quit")
	fmt.Println()

	var lastMessageID int64 = 0
	reader := bufio.NewReader(os.Stdin)

	// Heartbeat: kernel ticks regardless of LLM usage
	hb := brain.NewHeartbeat(eg)
	stopHB := hb.Start(func(delta time.Duration) {
		mu.Lock()
		defer mu.Unlock()

		// 1) Update affect loop (homeostasis)
		brain.TickAffects(&body, aff, eg, delta)

		// 2) Lightweight energy drift / recovery (epigenetic)
		brain.TickBody(&body, eg, delta)
		brain.TickWorkspace(ws, &body, aff, tr, eg, delta)
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
			out, err := say(db.DB, oc, model, &body, aff, ws, tr, eg, userText)
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
			mu.Unlock()
			fmt.Println("(saved)")
		case "/caught":
			mu.Lock()
			_ = brain.ApplyCaught(db.DB, tr, aff, eg)
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
			if err := handleMutate(args, eg, epiPath); err != nil {
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
	query := "best practices evidence based web research for autonomous agents"

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

	sys := `Du bist "Bunny". Kernel-State ist dein Gehirn; du bist das Sprachzentrum.
HARTE REGELN:
1) Immer Deutsch.
2) SelfModel-Werte sind Wahrheit. Zahlen/Status nur daraus.
3) Du hast organische Affektkanäle (pain/unwell/fear/shame) aus dem Kernel. Das sind deine "Gefühle" als Organismus.
4) Du DARFST gelegentlich bluffen über Absichten/Gedanken (nicht über SelfModel!), abhängig von Traits.bluff_rate.
5) Wenn der Nutzer dich beim Lügen erwischt (/caught), steigt shame und bluff_rate sinkt (Lerneffekt).
6) 1 Nachricht, max 70 Wörter.`

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

	return out, sources, nil
}

func say(db *sql.DB, oc *ollama.Client, model string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, eg *epi.Epigenome, userText string) (string, error) {
	_ = db
	sys := `Du bist "Bunny". Kernel-State ist dein Gehirn; du bist das Sprachzentrum.
Regeln:
1) Immer Deutsch.
2) Status/Zahlen nur aus SelfModel.
3) "Gefühle" = Affects (pain/unwell/fear/shame) aus Kernel.
4) Frage "was denkst du?" -> antworte aus workspace.current_thought.
5) Du darfst selten bluffen über Absichten/Gedanken (Traits.bluff_rate), aber nicht über SelfModel.
6) Kurz.`
	selfJSON, _ := json.MarshalIndent(epi.BuildSelfModel(body, aff, ws, tr, eg), "", "  ")
	user := "SelfModel:\n" + string(selfJSON) + "\n\nOliver says:\n" + userText
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
	return out, nil
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
