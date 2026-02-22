package main

import (
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/brain"
	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
	"frankenstein-v0/internal/websense"
)

// ExecuteTurn: single place where strategy becomes actual execution.
// This replaces "policy as prompt hint".
func ExecuteTurn(db *sql.DB, epiPath string, oc *ollama.Client, modelSpeaker, modelStance string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, error) {
	// UI commands (were previously only available in console loop).
	if ok, out := handleWebCommands(userText); ok {
		return out, nil
	}
	if ok, out := handleThoughtCommands(db, userText); ok {
		return out, nil
	}
	if ok, out := handleCodeCommands(db, oc, eg, userText); ok {
		return out, nil
	}
	if ok, out := handleABCommands(db, eg, userText); ok {
		return out, nil
	}
	if ok, out := handlePickCommand(db, userText); ok {
		return out, nil
	}
	// Natural confirmation: if last auto asked about materializing thought_proposals and user says "ja", show them.
	if isAffirmative(userText) && brain.CountThoughtProposals(db, "proposed") > 0 && lastAutoAsked(db, "ausformulieren", 10*time.Minute) {
		return brain.RenderThoughtProposalList(db, 10), nil
	}

	// Semantic memory (generic long-term facts): can answer/store before LLM.
	if ok, out := brain.SemanticMemoryStep(db, eg, userText); ok && strings.TrimSpace(out) != "" {
		return out, nil
	}

	// Topic should follow current user turn (no lock-in to previous topic).
	if ws != nil {
		t := brain.ExtractTopic(userText)
		if t != "" {
			ws.ActiveTopic = t
			ws.LastTopic = t
			brain.SaveActiveTopic(db, t)
			brain.BumpInterest(db, t, 0.10)
		}
	}

	if ws != nil && !ws.LLMAvailable {
		low := strings.ToLower(userText)
		if strings.Contains(low, "fühl") || strings.Contains(low, "wie geht") {
			return "Ich kann dir meinen Zustand nennen (Ressourcen/Drives), aber mein Sprachzentrum (LLM/Ollama) ist auf diesem Gerät gerade nicht verfügbar. Installier/Starte Ollama, dann kann ich normal antworten.", nil
		}
		return "LLM backend offline. Ich bin da, aber mein Sprachzentrum (LLM/Ollama) ist gerade offline. Willst du, dass ich dir helfe, Ollama zu installieren/zu starten?", nil
	}

	// --- Generic info gate (learned IDF) ---
	// Score first, then observe (avoid self-influencing DF during the same turn).
	low, info := brain.IsLowInfo(db, eg, userText)
	brain.ObserveUtterance(db, userText)
	if ws != nil {
		ws.LastUserInfoScore = info.Score
		ws.LastUserTopToken = info.TopToken
	}
	if low {
		// Low-information utterance: never research/stance/topic drift.
		// Generic conversational handshake.
		if ws != nil && ws.SurvivalMode {
			return "Ich bin da. Willst du einfach kurz plaudern oder ein konkretes Thema?", nil
		}
		return "Hi 🙂 Willst du einfach reden oder soll ich ein Thema vorschlagen?", nil
	}

	// --- Intent detection ---
	intent := brain.DetectIntentWithEpigenome(userText, eg)

	// --- A/B training mode (preference data for LoRA / behavior) ---
	// Notes:
	// - We skip EXTERNAL_FACT to avoid double websense runs.
	// - If A/B is enabled but cannot be produced (missing model, Ollama down, etc.),
	//   we return a clear diagnostic instead of silently falling back.
	if abEnabled(db) && intent != brain.IntentExternalFact {
		msg, ok := runABTrial(db, epiPath, oc, modelSpeaker, modelStance, body, aff, ws, tr, dr, eg, userText)
		if ok {
			return msg, nil
		}
		return "A/B Training ist AN, aber der Trial konnte nicht erzeugt werden.\n" +
			"Prüfe:\n" +
			"1) Existiert das B-Model wirklich? (Terminal: `ollama list`)\n" +
			"2) Läuft Ollama? (Terminal: `ollama ps` oder `curl http://localhost:11434/api/tags`)\n" +
			"3) Teste ohne LoRA: `/ab set b_model llama3.1:8b`", nil
	}

	intentMode := brain.IntentToMode(intent)

	survival := 0.0
	social := 0.0
	if ws != nil {
		survival = ws.DrivesEnergyDeficit
		social = ws.SocialCraving
	}

	brain.ApplySurvivalGate(ws, survival)

	topic := ""
	if ws != nil && ws.ActiveTopic != "" {
		topic = ws.ActiveTopic
	} else if ws != nil {
		topic = ws.LastTopic
	}
	topic = brain.NormalizeTopic(topic)

	ctxKey := brain.MakePolicyContext(intentMode, survival, social)
	choice := brain.ChoosePolicy(db, ctxKey)
	if ws != nil {
		ws.LastPolicyCtx = choice.ContextKey
		ws.LastPolicyAction = choice.Action
		ws.LastPolicyStyle = choice.Style
		ws.LastRoutedIntent = intentMode
	}
	brain.PlanFromAction(ws, topic, choice.Action)

	if ws != nil && ws.SurvivalMode && choice.Action == "research_then_answer" {
		choice.Action = "direct_answer"
		ws.LastPolicyAction = "direct_answer"
		brain.PlanFromAction(ws, topic, "direct_answer")
	}

	switch choice.Action {
	case "ask_clarify":
		if topic != "" {
			return "Kurze Rückfrage zum Thema \"" + topic + "\": Willst du Fakten/Status, eine Bewertung/Haltung, oder Optionen mit Trade-offs?", nil
		}
		return "Kurze Rückfrage: Willst du Fakten/Status, eine Bewertung/Haltung, oder Optionen mit Trade-offs?", nil
	case "social_ping":
		// For user turns, never return empty. If autonomy is blocked, fallback to direct answer.
		if ws != nil && !ws.AutonomyAllowed {
			choice.Action = "direct_answer"
			ws.LastPolicyAction = "direct_answer"
			return say(db, epiPath, oc, modelSpeaker, modelStance, body, aff, ws, tr, dr, eg, userText)
		}
		if topic != "" {
			return "Bevor ich weiterlaufe: soll ich beim Thema \"" + topic + "\" eher recherchieren, eine Haltung bilden, oder gemeinsam Optionen strukturieren?", nil
		}
		return "Soll ich dir gerade eher mit Fakten, einer Entscheidung oder einem Gedanken-Austausch helfen?", nil
	case "stance_then_answer":
		return answerWithStance(db, oc, modelStance, body, aff, ws, tr, eg, userText)
	case "research_then_answer":
		if ws != nil && !ws.WebAllowed {
			return "Ich würde dafür normalerweise kurz recherchieren, aber ich bin gerade im Ressourcen-Schonmodus. Gib mir bitte einen konkreten Aspekt oder eine Quelle, dann antworte ich kompakt.", nil
		}
		q := strings.TrimSpace(brain.NormalizeSearchQuery(userText))
		if q == "" {
			q = topic
		}
		return answerWithEvidence(db, oc, modelSpeaker, body, aff, ws, tr, eg, q)
	default:
		out, err := say(db, epiPath, oc, modelSpeaker, modelStance, body, aff, ws, tr, dr, eg, userText)
		if err == nil && strings.TrimSpace(out) == "" {
			return "Ich bin da. Sag mir kurz, was du von mir willst: Status, Meinung oder einfach reden?", nil
		}
		return out, err
	}
}

func handleCodeCommands(db *sql.DB, oc *ollama.Client, eg *epi.Epigenome, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/code") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) == 1 {
		return true, brain.RenderCodeProposalList(db, 20)
	}
	sub := strings.ToLower(strings.TrimSpace(parts[1]))
	switch sub {
	case "list":
		return true, brain.RenderCodeProposalList(db, 20)
	case "show":
		if len(parts) < 3 {
			return true, "Use: /code show <id>"
		}
		id := parseID(parts[2])
		return true, brain.RenderCodeProposal(db, id)
	case "draft":
		if len(parts) < 3 {
			return true, "Use: /code draft <id>"
		}
		id := parseID(parts[2])
		title, diffText, status, notes, ok := brain.GetCodeProposalFull(db, id)
		_ = status
		if !ok {
			return true, "Nicht gefunden."
		}
		spec := strings.TrimSpace(notes)
		if spec == "" {
			spec = strings.TrimSpace(diffText)
		}
		if spec == "" {
			return true, "Kein Inhalt vorhanden (notes+diff leer)."
		}
		coder := eg.ModelFor("coder", eg.ModelFor("speaker", "llama3.1:8b"))
		ctx := codeIndexContext(db, title, spec)
		sys := "Du bist ein Go-Engineer. Gib NUR einen unified diff aus (git apply kompatibel). " +
			"Keine Erklärungen. Minimaler Patch. Pfade relativ zum Repo-Root. " +
			"Nur in cmd/ oder internal/ ändern. Kein go.mod/go.sum. " +
			"Wenn möglich: Tests hinzufügen."
		user := "GOAL/TITLE:\n" + title + "\n\nSPEC/NOTES:\n" + spec + "\n\nCODE_INDEX_CONTEXT:\n" + ctx
		out, err := oc.Chat(coder, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
		if err != nil {
			return true, "LLM draft failed: " + err.Error()
		}
		out = strings.TrimSpace(out)
		if !strings.Contains(out, "diff --git") {
			return true, "LLM hat keinen unified diff geliefert. (Erwartet: diff --git ...)"
		}
		if bad := firstDisallowedPath(out); bad != "" {
			return true, "Diff enthält disallowed path: " + bad
		}
		brain.UpdateCodeProposal(db, id, out, "proposed")
		return true, "OK. Diff erzeugt und in code_proposal #" + strconv.FormatInt(id, 10) + " gespeichert.\nWeiter: /code apply " + strconv.FormatInt(id, 10)
	case "apply":
		if len(parts) < 3 {
			return true, "Use: /code apply <id>"
		}
		id := parseID(parts[2])
		_, diffText, _, _, ok := brain.GetCodeProposalFull(db, id)
		if !ok {
			return true, "Nicht gefunden."
		}
		diffText = strings.TrimSpace(diffText)
		if !strings.Contains(diffText, "diff --git") {
			return true, "In code_proposal #" + strconv.FormatInt(id, 10) + " ist noch kein unified diff. Erst: /code draft " + strconv.FormatInt(id, 10)
		}
		if bad := firstDisallowedPath(diffText); bad != "" {
			return true, "Diff enthält disallowed path: " + bad
		}
		msg, err := applyPatchInRepo(diffText)
		if err != nil {
			return true, "Apply fehlgeschlagen: " + err.Error() + "\n" + msg
		}
		brain.MarkCodeProposal(db, id, "applied")
		return true, "OK. Patch angewendet + go test ./... OK. (code_proposal #" + strconv.FormatInt(id, 10) + " → applied)"
	case "reject":
		if len(parts) < 3 {
			return true, "Use: /code reject <id>"
		}
		id := parseID(parts[2])
		brain.MarkCodeProposal(db, id, "rejected")
		return true, "OK. code_proposal #" + strconv.FormatInt(id, 10) + " → rejected"
	default:
		return true, "Use: /code list | /code show <id> | /code draft <id> | /code apply <id> | /code reject <id>"
	}
}

func codeIndexContext(db *sql.DB, title, spec string) string {
	if db == nil {
		return ""
	}
	q := strings.ToLower(title + " " + spec)
	keys := []string{}
	for _, t := range brain.TokenizeAlphaNumLower(q) {
		if len(t) < 4 {
			continue
		}
		if _, bad := map[string]struct{}{"topic": {}, "drift": {}, "fix": {}, "prevent": {}}[t]; bad {
			continue
		}
		keys = append(keys, t)
		if len(keys) >= 3 {
			break
		}
	}
	if len(keys) == 0 {
		keys = []string{"topic", "gate", "workspace"}
	}
	like := "%" + keys[0] + "%"
	rows, err := db.Query(`SELECT path, summary FROM code_index WHERE lower(path) LIKE ? OR lower(summary) LIKE ? LIMIT 14`, like, like)
	if err != nil {
		return ""
	}
	defer rows.Close()
	var b strings.Builder
	for rows.Next() {
		var p, s string
		_ = rows.Scan(&p, &s)
		b.WriteString("- " + strings.TrimSpace(p) + ": " + strings.TrimSpace(s) + "\n")
	}
	return strings.TrimSpace(b.String())
}

func firstDisallowedPath(diff string) string {
	lines := strings.Split(diff, "\n")
	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if !strings.HasPrefix(ln, "diff --git ") {
			continue
		}
		// diff --git a/<p> b/<p>
		parts := strings.Fields(ln)
		if len(parts) < 4 {
			continue
		}
		a := strings.TrimPrefix(parts[2], "a/")
		b := strings.TrimPrefix(parts[3], "b/")
		for _, p := range []string{a, b} {
			p = strings.TrimSpace(p)
			if p == "" {
				continue
			}
			if strings.HasPrefix(p, "cmd/") || strings.HasPrefix(p, "internal/") {
				// ok
				continue
			}
			return p
		}
	}
	// forbid go.mod/go.sum anywhere
	if strings.Contains(diff, "go.mod") || strings.Contains(diff, "go.sum") {
		return "go.mod/go.sum"
	}
	return ""
}

func applyPatchInRepo(diff string) (string, error) {
	diff = strings.TrimSpace(diff)
	if diff == "" {
		return "", fmt.Errorf("empty diff")
	}
	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("bunny_patch_%d.diff", time.Now().UnixNano()))
	_ = os.WriteFile(tmp, []byte(diff), 0644)
	defer os.Remove(tmp)

	// ensure tools exist
	if _, err := exec.LookPath("git"); err != nil {
		return "", fmt.Errorf("git not found in PATH")
	}
	if _, err := exec.LookPath("go"); err != nil {
		return "", fmt.Errorf("go not found in PATH")
	}

	repo, err := gitRepoRoot()
	if err != nil {
		return "", err
	}

	var log strings.Builder
	log.WriteString("[code apply]\n")
	log.WriteString("repo: " + repo + "\n")

	// require clean working tree
	status, err := runCmdDir(repo, "git", "status", "--porcelain")
	if err != nil {
		log.WriteString("git status failed:\n" + status + "\n")
		return strings.TrimSpace(log.String()), err
	}
	if strings.TrimSpace(status) != "" {
		log.WriteString("working tree NOT clean:\n" + status + "\n")
		return strings.TrimSpace(log.String()), fmt.Errorf("working tree not clean (commit/stash first)")
	}

	log.WriteString("1) git apply --check\n")
	out, err := runCmdDir(repo, "git", "apply", "--check", tmp)
	if err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), err
	}

	log.WriteString("2) git apply\n")
	out, err = runCmdDir(repo, "git", "apply", tmp)
	if err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), err
	}

	log.WriteString("3) go test ./...\n")
	testOut, testErr := runCmdDir(repo, "go", "test", "./...")
	if testErr != nil {
		log.WriteString("go test FAILED:\n" + testOut + "\n")
		rb, _ := runCmdDir(repo, "git", "apply", "-R", tmp)
		log.WriteString("rollback:\n" + rb + "\n")
		return strings.TrimSpace(log.String()), fmt.Errorf("go test failed; patch rolled back")
	}

	if strings.TrimSpace(testOut) != "" {
		log.WriteString(testOut + "\n")
	}
	log.WriteString("OK\n")
	return strings.TrimSpace(log.String()), nil
}

func gitRepoRoot() (string, error) {
	// Optional override (useful if bunny is started outside the repo).
	if v := strings.TrimSpace(os.Getenv("BUNNY_REPO_ROOT")); v != "" {
		return v, nil
	}
	// Use current working dir, but ask git for actual root.
	out, err := runCmdDir("", "git", "rev-parse", "--show-toplevel")
	if err != nil {
		if strings.TrimSpace(out) == "" {
			out = err.Error()
		}
		return "", fmt.Errorf("cannot determine git repo root (are you running bunny inside the repo?): %s", strings.TrimSpace(out))
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return "", fmt.Errorf("cannot determine git repo root (empty)")
	}
	return out, nil
}

func runCmdDir(dir string, bin string, args ...string) (string, error) {
	cmd := exec.Command(bin, args...)
	if strings.TrimSpace(dir) != "" {
		cmd.Dir = dir
	}
	out, err := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)), err
}

func handleABCommands(db *sql.DB, eg *epi.Epigenome, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/ab") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) == 1 {
		return true, renderABStatus(db, eg)
	}
	sub := strings.ToLower(parts[1])
	switch sub {
	case "on":
		kvSet(db, "ab_enabled", "1")
		return true, "OK. A/B Training ist AN.\n" + renderABStatus(db, eg)
	case "off":
		kvSet(db, "ab_enabled", "0")
		return true, "OK. A/B Training ist AUS."
	case "status":
		return true, renderABStatus(db, eg)
	case "set":
		if len(parts) < 4 {
			return true, "Use: /ab set a_model|b_model|a_style|b_style <value>"
		}
		k := strings.ToLower(strings.TrimSpace(parts[2]))
		v := strings.TrimSpace(strings.TrimPrefix(line, parts[0]+" "+parts[1]+" "+parts[2]))
		v = strings.TrimSpace(v)
		if v == "" {
			return true, "Value leer."
		}
		allowed := map[string]string{"a_model": "ab_model_a", "b_model": "ab_model_b", "a_style": "ab_style_a", "b_style": "ab_style_b"}
		key, ok := allowed[k]
		if !ok {
			return true, "Use: /ab set a_model|b_model|a_style|b_style <value>"
		}
		kvSet(db, key, v)
		return true, "OK. " + k + " gesetzt.\n" + renderABStatus(db, eg)
	default:
		return true, "Use: /ab on|off|status|set ..."
	}
}

func handlePickCommand(db *sql.DB, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/pick") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) < 3 {
		return true, "Use: /pick <ab_id> a|b|none"
	}
	id := parseID(parts[1])
	choice := strings.ToLower(strings.TrimSpace(parts[2]))
	t, ok := brain.GetABTrial(db, id)
	if !ok {
		return true, "AB# nicht gefunden."
	}
	if t.Status == "chosen" {
		// already chosen → return the chosen answer
		switch t.Choice {
		case "a":
			return true, t.AText
		case "b":
			return true, t.BText
		default:
			return true, "OK. (none)"
		}
	}
	if err := brain.ChooseABTrial(db, id, choice); err != nil {
		return true, "ERR: " + err.Error()
	}
	switch choice {
	case "a":
		return true, t.AText
	case "b":
		return true, t.BText
	default:
		return true, "OK. (none)"
	}
}

func abEnabled(db *sql.DB) bool {
	return strings.TrimSpace(kvGet(db, "ab_enabled")) == "1"
}

func renderABStatus(db *sql.DB, eg *epi.Epigenome) string {
	aModel := strings.TrimSpace(kvGet(db, "ab_model_a"))
	bModel := strings.TrimSpace(kvGet(db, "ab_model_b"))
	aStyle := strings.TrimSpace(kvGet(db, "ab_style_a"))
	bStyle := strings.TrimSpace(kvGet(db, "ab_style_b"))
	if aModel == "" {
		aModel = eg.ModelFor("speaker", "llama3.1:8b")
	}
	if bStyle == "" {
		bStyle = "empathisch-direkt"
	}
	en := "OFF"
	if abEnabled(db) {
		en = "ON"
	}
	return "A/B Training: " + en + "\n" +
		"A model: " + aModel + "\n" +
		"B model: " + pickNonEmpty(bModel, "<unset>") + "\n" +
		"A style: " + pickNonEmpty(aStyle, "default") + "\n" +
		"B style: " + bStyle + "\n\n" +
		"Setze B-Model z.B.: /ab set b_model bunny-speaker-lora-v1"
}

func pickNonEmpty(v, fb string) string {
	if strings.TrimSpace(v) != "" {
		return strings.TrimSpace(v)
	}
	return fb
}

func kvGet(db *sql.DB, key string) string {
	if db == nil {
		return ""
	}
	var v string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, key).Scan(&v)
	return strings.TrimSpace(v)
}

func kvSet(db *sql.DB, key, val string) {
	if db == nil {
		return
	}
	_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?)
		ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		strings.TrimSpace(key), strings.TrimSpace(val), time.Now().Format(time.RFC3339))
}

func runABTrial(db *sql.DB, epiPath string, oc *ollama.Client, modelSpeaker, modelStance string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, bool) {
	bModel := strings.TrimSpace(kvGet(db, "ab_model_b"))
	if bModel == "" {
		return "A/B Training ist AN, aber B-Model ist nicht gesetzt.\nNutze: /ab set b_model <modelName>", true
	}
	aModel := strings.TrimSpace(kvGet(db, "ab_model_a"))
	if aModel == "" {
		aModel = eg.ModelFor("speaker", modelSpeaker)
	}
	aStyle := strings.TrimSpace(kvGet(db, "ab_style_a"))
	bStyle := strings.TrimSpace(kvGet(db, "ab_style_b"))
	if bStyle == "" {
		bStyle = "STYLE: empathisch-direkt. Kurz, warm, direkt. Keine unnötigen Rückfragen. 2–5 Sätze."
	}

	// Clone state to avoid double side-effects.
	bodyA := *body
	bodyB := *body
	wsA := cloneWorkspace(ws)
	wsB := cloneWorkspace(ws)
	affA := cloneAffect(aff)
	affB := cloneAffect(aff)
	drA := *dr
	drB := *dr

	aOut, errA := sayWithMutation(db, epiPath, oc, aModel, modelStance, &bodyA, affA, wsA, tr, &drA, eg, userText, aStyle)
	bOut, errB := sayWithMutation(db, epiPath, oc, bModel, modelStance, &bodyB, affB, wsB, tr, &drB, eg, userText, bStyle)
	aOut = strings.TrimSpace(aOut)
	bOut = strings.TrimSpace(bOut)

	// IMPORTANT: Do not silently fall back.
	if errA != nil || errB != nil {
		return "A/B Trial konnte nicht erzeugt werden (LLM Fehler).\n" +
				"A (" + aModel + "): " + errString(errA) + "\n" +
				"B (" + bModel + "): " + errString(errB) + "\n\n" +
				"Tipp: Prüfe Modelnamen mit `ollama list`. Falls dein LoRA-Model fehlt, baue es per `ollama create <name> -f Modelfile`.",
			true
	}
	if aOut == "" || bOut == "" {
		return "A/B Trial konnte nicht erzeugt werden (leere Kandidaten).\n" +
				"A (" + aModel + ") len=" + strconv.Itoa(len(aOut)) + "\n" +
				"B (" + bModel + ") len=" + strconv.Itoa(len(bOut)) + "\n\n" +
				"Tipp: Teste B-Model erst als normales Model: `/ab set b_model llama3.1:8b`.",
			true
	}

	id, err := brain.InsertABTrial(db, userText, aModel, aOut, bModel, bOut)
	if err != nil {
		return "ERR: " + err.Error(), true
	}
	t, _ := brain.GetABTrial(db, id)
	return brain.RenderABTrial(t), true
}
func parseID(raw string) int64 {
	raw = strings.TrimSpace(raw)
	raw = strings.TrimPrefix(raw, "#")
	id, _ := strconv.ParseInt(raw, 10, 64)
	return id
}

func errString(err error) string {
	if err == nil {
		return "<nil>"
	}
	s := strings.TrimSpace(err.Error())
	if s == "" {
		return "<error>"
	}
	return s
}

func cloneWorkspace(ws *brain.Workspace) *brain.Workspace {
	if ws == nil {
		return nil
	}
	c := *ws
	if ws.PlanSteps != nil {
		c.PlanSteps = append([]string{}, ws.PlanSteps...)
	}
	return &c
}

func cloneAffect(a *brain.AffectState) *brain.AffectState {
	if a == nil {
		return nil
	}
	c := brain.NewAffectState()
	for _, k := range a.Keys() {
		c.Set(k, a.Get(k))
	}
	return c
}

func isAffirmative(s string) bool {
	t := strings.TrimSpace(strings.ToLower(s))
	switch t {
	case "ja", "j", "jo", "yes", "y", "ok", "okay", "klar", "bitte", "gerne", "mach", "mach das", "mach es", "genau":
		return true
	default:
		return false
	}
}

func lastAutoAsked(db *sql.DB, contains string, within time.Duration) bool {
	if db == nil {
		return false
	}
	var txt string
	var ts string
	_ = db.QueryRow(`SELECT m.text, m.created_at
		FROM messages m
		JOIN message_meta mm ON mm.message_id=m.id
		WHERE mm.kind='auto'
		ORDER BY m.id DESC LIMIT 1`).Scan(&txt, &ts)
	txt = strings.ToLower(strings.TrimSpace(txt))
	if txt == "" || !strings.Contains(txt, strings.ToLower(contains)) {
		return false
	}
	tm, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return false
	}
	return time.Since(tm) <= within
}

func handleWebCommands(userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/web") && !strings.HasPrefix(line, "/websense") {
		return false, ""
	}
	// Usage:
	// /web test <query>
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return true, "Use: /web test <query>"
	}
	if parts[1] != "test" {
		return true, "Use: /web test <query>"
	}
	q := strings.TrimSpace(strings.TrimPrefix(line, parts[0]+" "+parts[1]))
	q = strings.TrimSpace(q)
	if q == "" {
		return true, "Use: /web test <query>"
	}
	results, err := websense.Search(q, 6)
	if err != nil || len(results) == 0 {
		if err != nil {
			return true, "websense.Search failed: " + err.Error()
		}
		return true, "Keine Ergebnisse."
	}
	var b strings.Builder
	b.WriteString("websense.Search OK. Top Ergebnisse:\n")
	for i := 0; i < len(results) && i < 5; i++ {
		title := strings.TrimSpace(results[i].Title)
		u := strings.TrimSpace(results[i].URL)
		sn := strings.TrimSpace(results[i].Snippet)
		if len(sn) > 140 {
			sn = sn[:140] + "..."
		}
		b.WriteString("- " + title + "\n  " + u + "\n  " + sn + "\n")
	}
	return true, strings.TrimSpace(b.String())
}
func handleThoughtCommands(db *sql.DB, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/thought") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) == 1 {
		return true, brain.RenderThoughtProposalList(db, 10)
	}
	switch parts[1] {
	case "list":
		return true, brain.RenderThoughtProposalList(db, 10)
	case "show":
		if len(parts) < 3 {
			return true, "Use: /thought show <id>"
		}
		id := parseID(parts[2])
		return true, brain.RenderThoughtProposal(db, id)
	case "materialize":
		if len(parts) < 3 {
			return true, "Use: /thought materialize <id|all>"
		}
		arg := strings.ToLower(strings.TrimSpace(parts[2]))
		if arg == "all" {
			return true, brain.MaterializeAllThoughtProposals(db, 25)
		}
		id := parseID(arg)
		msg, _ := brain.MaterializeThoughtProposal(db, id)
		return true, msg
	default:
		return true, "Use: /thought list | /thought show <id> | /thought materialize <id|all>"
	}
}
