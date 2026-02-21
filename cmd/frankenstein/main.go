package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

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
	URL      string `json:"url"`
	Domain   string `json:"domain"`
	Title    string `json:"title"`
	Snippet  string `json:"snippet"`
	FetchedAt string `json:"fetched_at"`
	Hash     string `json:"hash"`
}

func main() {
	model := getenv("FRANK_MODEL", "llama3.1:8b")
	ollamaURL := getenv("OLLAMA_URL", "http://localhost:11434")
	dbPath := getenv("FRANK_DB", "data/frankenstein.sqlite")

	_ = os.MkdirAll("data", 0o755)

	db, err := state.Open(dbPath)
	must(err)
	defer db.Close()

	oc := ollama.New(ollamaURL)

	// v0 BodyState
	body := BodyState{
		Energy:        80,
		MemLoad:       0,
		WebCountHour:  0,
		CooldownUntil: time.Time{},
	}

	fmt.Println("Frankenstein v0 online.")
	fmt.Println("Commands: /think | /say <text> | /rate <up|meh|down> | /quit")
	fmt.Println()

	var lastMessageID int64 = 0

	for {
		fmt.Print("> ")
		var line string
		_, _ = fmt.Scanln(&line)

		// Scanln splits by space; re-read full line via stdin buffer is annoying.
		// For v0: accept simple commands; if you want full lines, we can switch to bufio.Reader next.
		if line == "/quit" {
			return
		}

		switch line {
		case "/think":
			// one “idle” cycle: pick a topic, search, fetch, propose a message
			msgText, sources, err := oneThinkCycle(db.DB, oc, model, &body)
			if err != nil {
				fmt.Println("ERR:", err)
				continue
			}
			if msgText == "" {
				fmt.Println("(silent)")
				continue
			}
			if time.Now().Before(body.CooldownUntil) {
				fmt.Println("(cooldown, message queued but not spoken)")
				continue
			}
			lastMessageID = persistMessage(db.DB, msgText, sources, 0.5)
			fmt.Println()
			fmt.Println("Frankenstein:", msgText)
			fmt.Println()
			fmt.Println("Rate it with: /rate up | /rate meh | /rate down")
		case "/say":
			fmt.Println("Use: /say hello (single token for v0). We'll improve input handling next.")
		case "/rate":
			fmt.Println("Use: /rate up|meh|down (single token for v0).")
		default:
			// handle /rate <x> quickly with env-like hack: read second token from args
			// v0 limitation: rely on os.Args? We'll patch next iteration.
			fmt.Println("Unknown. Try /think or /quit.")
		}

		// NOTE: We'll replace the crude Scanln loop with bufio in the next patch so /say works properly.
		_ = lastMessageID
	}
}

func oneThinkCycle(db *sql.DB, oc *ollama.Client, model string, body *BodyState) (string, []SourceRecord, error) {
	// Minimal: fixed curiosity query for the first run.
	// Next iteration we’ll generate topics from interests + memory.
	query := "best practices evidence based web research for autonomous agents"

	// Web cost model
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

	// Fetch top 1–2
	var sources []SourceRecord
	for i := 0; i < len(results) && i < 2; i++ {
		fr, err := websense.Fetch(results[i].URL)
		if err != nil {
			continue
		}
		storeSource(db, fr)
		sources = append(sources, SourceRecord{
			URL:      fr.URL,
			Domain:   fr.Domain,
			Title:    fr.Title,
			Snippet:  fr.Snippet,
			FetchedAt: fr.FetchedAt.Format(time.RFC3339),
			Hash:     fr.Hash,
		})
	}

	if len(sources) == 0 {
		return "", nil, nil
	}

	// Ask LLM to synthesize a single proactive message with evidence mention.
	sys := `You are "Frankenstein", an embodied research organism.
Rules:
- Be concise and non-spammy.
- Only claim "I read on <domain> ..." if sources are provided.
- Produce ONE message to Oliver: a claim + why it matters + one question.
- Max 70 words.`
	srcJSON, _ := json.MarshalIndent(sources, "", "  ")

	user := "BodyState:\n" + fmt.Sprintf("energy=%.1f, webCountHour=%d\n", body.Energy, body.WebCountHour) +
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

	// Communication cost + cooldown
	body.Energy -= 1.5
	body.CooldownUntil = time.Now().Add(2 * time.Minute)

	return out, sources, nil
}

func storeSource(db *sql.DB, fr *websense.FetchResult) {
	_, _ = db.Exec(
		`INSERT INTO sources(url, domain, title, fetched_at, content_hash, snippet) VALUES(?,?,?,?,?,?)`,
		fr.URL, fr.Domain, fr.Title, fr.FetchedAt.Format(time.RFC3339), fr.Hash, fr.Snippet,
	)
}

func persistMessage(db *sql.DB, text string, sources []SourceRecord, priority float64) int64 {
	b, _ := json.Marshal(sources)
	res, err := db.Exec(
		`INSERT INTO messages(created_at, priority, text, sources_json) VALUES(?,?,?,?)`,
		time.Now().Format(time.RFC3339), priority, text, string(b),
	)
	if err != nil {
		return 0
	}
	id, _ := res.LastInsertId()
	return id
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
