package main

import (
	"database/sql"
	"encoding/json"
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
	if ok, out := handleEpiCommands(db, epiPath, eg, userText); ok {
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
	if isStopProposalSpam(userText) {
		brain.UpdatePreferenceEMA(db, "auto:proposal_pings", -1.0, 0.25)
		brain.UpdatePreferenceEMA(db, "auto:thought_pings", -1.0, 0.25)
		brain.UpdatePreferenceEMA(db, "auto:proposal_engine_announce", -1.0, 0.25)
		return "Verstanden. Ich pinge dich mit Selbstverbesserungs-Ideen nur noch sehr sparsam. Wenn du sie sehen willst: /thought list (oder /code list).", nil
	}

	// Semantic memory (generic long-term facts): can answer/store before LLM.
	if ok, out := brain.SemanticMemoryStep(db, eg, userText); ok && strings.TrimSpace(out) != "" {
		return out, nil
	}

	// Topic update is conservative to avoid accidental drift on follow-up questions.
	if ws != nil {
		t := brain.UpdateActiveTopic(ws, userText)
		if t != "" {
			ws.ActiveTopic = t
			ws.LastTopic = t
			brain.SaveActiveTopic(db, t)
			brain.BumpInterest(db, t, 0.10)
		}
	}

	if ws != nil && !ws.LLMAvailable {
		// LLM offline: respond via epigenetic reflex templates (no keyword hacks in code).
		return brain.OfflineReflexReply(eg, ws, userText), nil
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

	// --- Survival gate (kernel truth) should be applied BEFORE routing/training ---
	survival := 0.0
	social := 0.0
	if ws != nil {
		survival = ws.DrivesEnergyDeficit
		social = ws.SocialCraving
		brain.ApplySurvivalGate(ws, survival)
	}

	// --- Intent detection (hybrid: epigenome rules + NB fallback) ---
	nb := brain.NewNBIntent(db)
	intent := brain.DetectIntentHybrid(userText, eg, nb)
	intentMode := brain.IntentToMode(intent)

	// --- Cortex sensor-gate: decide if WebSense is required ---
	gateModel := eg.ModelFor("scout", eg.ModelFor("speaker", modelSpeaker))
	rd := brain.DecideResearchCortex(db, oc, gateModel, userText, intent, ws, tr, dr, aff)
	if ws != nil {
		ws.LastSenseNeedWeb = rd.Do
		ws.LastSenseScore = rd.Score
		ws.LastSenseQuery = rd.Query
		ws.LastSenseReason = rd.Reason
		ws.LastSenseText = userText
	}

	// --- A/B training mode (preference data for LoRA / behavior) ---
	// Notes:
	// - We skip EXTERNAL_FACT to avoid double websense runs.
	// - We also skip if cortex says research is needed (avoid training on hallucinations).
	// - If training is enabled but cannot be produced (missing model, Ollama down, etc.),
	//   we return a clear diagnostic instead of silently falling back.
	if trainEnabled(db) && intent != brain.IntentExternalFact && !rd.Do {
		msg, ok := runTrainTrial(db, epiPath, oc, modelSpeaker, modelStance, body, aff, ws, tr, dr, eg, userText)
		if ok {
			return msg, nil
		}
		return "A/B Training ist AN, aber der Trial konnte nicht erzeugt werden.\n" +
			"Prüfe:\n" +
			"1) Existiert das B-Model wirklich? (Terminal: `ollama list`)\n" +
			"2) Läuft Ollama? (Terminal: `ollama ps` oder `curl http://localhost:11434/api/tags`)\n" +
			"3) Teste ohne LoRA: `/ab set b_model llama3.1:8b`", nil
	}

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
	// Truth-gate: if cortex decided web is needed, avoid direct_answer bluffing.
	if rd.Do && ws != nil && ws.WebAllowed && choice.Action == "direct_answer" {
		choice.Action = "research_then_answer"
		ws.LastPolicyAction = choice.Action
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
		if strings.TrimSpace(rd.Query) != "" {
			q = strings.TrimSpace(rd.Query)
		}
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
		coder := selectCoderModel(oc, eg)
		ctx := codeIndexContext(db, title, spec)
		sys := "Du bist ein Go-Engineer. Gib NUR einen unified diff aus (git apply kompatibel). " +
			"Keine Erklärungen. Minimaler Patch. Pfade relativ zum Repo-Root. " +
			"Nur in cmd/ oder internal/ ändern. Kein go.mod/go.sum. " +
			"Wenn möglich: Tests hinzufügen. " +
			"WICHTIG: In jedem Hunk muss JEDE Zeile mit genau einem Prefix beginnen: ' ', '+', '-', oder '\\\\'."
		user := "GOAL/TITLE:\n" + title + "\n\nSPEC/NOTES:\n" + spec + "\n\nCODE_INDEX_CONTEXT:\n" + ctx
		out, err := oc.Chat(coder, []ollama.Message{{Role: "system", Content: sys}, {Role: "user", Content: user}})
		if err != nil {
			return true, "LLM draft failed: " + err.Error()
		}

		// 1) sanitize / strip fences
		out = stripCodeFences(strings.TrimSpace(out))

		if retryPrompt := draftDiffRetryPrompt(out); strings.TrimSpace(retryPrompt) != "" {
			fixOut, fixErr := oc.Chat(coder, []ollama.Message{
				{Role: "system", Content: sys},
				{Role: "user", Content: user},
				{Role: "assistant", Content: out},
				{Role: "user", Content: retryPrompt},
			})
			if fixErr == nil {
				out = stripCodeFences(strings.TrimSpace(fixOut))
			}
		}

		// 2) basic validation
		out = normalizeUnifiedDiffHunks(out)
		if !strings.Contains(out, "diff --git") {
			return true, "LLM hat keinen unified diff geliefert. (Erwartet: diff --git ...)"
		}
		if bad := firstDisallowedPath(out); bad != "" {
			return true, "Diff enthält disallowed path: " + bad
		}
		if err := validateUnifiedDiffSyntax(out); err != nil {
			return true, "Diff syntaktisch ungültig: " + err.Error()
		}

		// 3) path guard: only existing files (or explicit new files) in repo / code_index
		repo, err := gitRepoRoot()
		if err != nil {
			return true, "Kann Repo-Root nicht bestimmen: " + err.Error() + "\nTipp: setze BUNNY_REPO_ROOT."
		}
		warn, err := validateDiffTouchedPaths(db, repo, out)
		if err != nil {
			return true, "Diff-Pfad-Check fehlgeschlagen: " + err.Error() + "\nTipp: /selfcode index ausführen."
		}

		// 4) preflight: git apply --check + go test in temporary worktree (compile gate before apply)
		log, err := preflightApplyAndTest(repo, out)
		if err != nil {
			return true, "Preflight fehlgeschlagen (Patch wird NICHT gespeichert):\n" + log
		}
		brain.UpdateCodeProposal(db, id, out, "proposed")
		msg := "OK. Diff validiert und compilierbar (go test OK) — in code_proposal #" + strconv.FormatInt(id, 10) + " gespeichert.\nWeiter: /code apply " + strconv.FormatInt(id, 10)
		if strings.TrimSpace(warn) != "" {
			msg += "\n" + warn
		}
		return true, msg
	case "apply":
		if len(parts) < 3 {
			return true, "Use: /code apply <id>"
		}
		id := parseID(parts[2])
		title, diffText, _, _, ok := brain.GetCodeProposalFull(db, id)
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
		msg, err := applyPatchInRepo(id, title, diffText)
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

func stripCodeFences(s string) string {
	s = strings.TrimSpace(s)
	if !strings.Contains(s, "```") {
		return s
	}
	// Take the first fenced block if present.
	i := strings.Index(s, "```")
	if i < 0 {
		return s
	}
	rest := s[i+3:]
	// skip optional language id line
	if j := strings.Index(rest, "\n"); j >= 0 {
		rest = rest[j+1:]
	}
	k := strings.Index(rest, "```")
	if k < 0 {
		return strings.TrimSpace(rest)
	}
	return strings.TrimSpace(rest[:k])
}

func draftDiffRetryPrompt(diff string) string {
	d := strings.TrimSpace(diff)
	if d == "" {
		return "Du hast keinen Diff geliefert. Bitte liefere JETZT nur einen vollständigen unified diff ab `diff --git ...` ohne Erklärungen."
	}
	if !strings.Contains(d, "diff --git") {
		return "Das Format ist falsch. Bitte liefere NUR einen unified diff im Format `diff --git a/... b/...` ohne Zusatztext."
	}
	if err := validateUnifiedDiffSyntax(d); err != nil {
		return "Dein Diff ist syntaktisch ungültig (" + err.Error() + "). Bitte gib den kompletten Diff erneut aus."
	}
	return ""
}

func selectCoderModel(oc *ollama.Client, eg *epi.Epigenome) string {
	fallback := eg.ModelFor("coder", eg.ModelFor("speaker", "llama3.1:8b"))
	if oc == nil {
		return fallback
	}
	models, err := oc.ListModels()
	if err != nil || len(models) == 0 {
		return fallback
	}
	candidates := []string{
		eg.ModelFor("coder", ""),
		"qwen2.5-coder:7b",
		"deepseek-coder:6.7b",
		"starcoder2:7b",
		fallback,
	}
	for _, m := range candidates {
		m = strings.TrimSpace(m)
		if m == "" {
			continue
		}
		if _, ok := models[m]; ok {
			return m
		}
	}
	return fallback
}

func validateUnifiedDiffSyntax(diff string) error {
	diff = normalizeUnifiedDiffHunks(diff)
	lines := strings.Split(diff, "\n")
	inHunk := false
	for i, ln := range lines {
		if strings.HasPrefix(ln, "diff --git ") {
			inHunk = false
			continue
		}
		if strings.HasPrefix(ln, "@@") {
			inHunk = true
			continue
		}
		if !inHunk {
			continue
		}
		if ln == "" {
			return fmt.Errorf("hunk line %d has empty prefix (expected ' ', '+', '-' or '\\\\')", i+1)
		}
		switch ln[0] {
		case ' ', '+', '-', '\\':
			// ok
		default:
			return fmt.Errorf("hunk line %d has invalid prefix %q", i+1, string(ln[0]))
		}
	}
	return nil
}

func normalizeUnifiedDiffHunks(diff string) string {
	lines := strings.Split(diff, "\n")
	inHunk := false
	for i, ln := range lines {
		if strings.HasPrefix(ln, "diff --git ") || strings.HasPrefix(ln, "--- ") || strings.HasPrefix(ln, "+++ ") || strings.HasPrefix(ln, "index ") {
			inHunk = false
			continue
		}
		if strings.HasPrefix(ln, "@@") {
			inHunk = true
			continue
		}
		if !inHunk {
			continue
		}
		if ln == "" {
			lines[i] = " "
		}
	}
	return strings.Join(lines, "\n")
}

type touchedFile struct {
	Path  string
	IsNew bool
}

func parseTouchedFiles(diff string) []touchedFile {
	lines := strings.Split(diff, "\n")
	var out []touchedFile
	for _, ln := range lines {
		if strings.HasPrefix(ln, "diff --git ") {
			parts := strings.Fields(ln)
			if len(parts) >= 4 {
				p := strings.TrimSpace(strings.TrimPrefix(parts[3], "b/"))
				out = append(out, touchedFile{Path: p})
			}
			continue
		}
		if strings.HasPrefix(ln, "new file mode") || strings.HasPrefix(ln, "--- /dev/null") {
			if len(out) > 0 {
				out[len(out)-1].IsNew = true
			}
		}
	}
	// dedupe by path while preserving order
	seen := map[string]bool{}
	result := make([]touchedFile, 0, len(out))
	for _, f := range out {
		if f.Path == "" || seen[f.Path] {
			continue
		}
		seen[f.Path] = true
		result = append(result, f)
	}
	return result
}

func codeIndexRowCount(db *sql.DB) int {
	if db == nil {
		return 0
	}
	var n int
	_ = db.QueryRow(`SELECT COUNT(*) FROM code_index`).Scan(&n)
	return n
}

func codeIndexHasPath(db *sql.DB, path string) bool {
	if db == nil {
		return false
	}
	var one int
	err := db.QueryRow(`SELECT 1 FROM code_index WHERE path=? LIMIT 1`, path).Scan(&one)
	return err == nil && one == 1
}

func validateDiffTouchedPaths(db *sql.DB, repoRoot string, diff string) (string, error) {
	files := parseTouchedFiles(diff)
	if len(files) == 0 {
		return "", fmt.Errorf("diff enthält keine erkennbaren File-Changes (diff --git ...)")
	}
	idxN := codeIndexRowCount(db)
	for _, f := range files {
		p := strings.TrimSpace(f.Path)
		if p == "" {
			continue
		}
		if strings.Contains(p, "..") {
			return "", fmt.Errorf("path traversal nicht erlaubt: %s", p)
		}
		if !(strings.HasPrefix(p, "cmd/") || strings.HasPrefix(p, "internal/")) {
			return "", fmt.Errorf("disallowed path root: %s", p)
		}
		abs := filepath.Join(repoRoot, filepath.FromSlash(p))
		_, err := os.Stat(abs)
		if err != nil && !f.IsNew {
			return "", fmt.Errorf("file nicht gefunden im Repo: %s", p)
		}
		if idxN > 0 && !f.IsNew && !codeIndexHasPath(db, p) {
			return "", fmt.Errorf("file nicht im code_index (run /selfcode index): %s", p)
		}
	}
	if idxN == 0 {
		return "WARN: code_index ist leer. Für strengere Guards: /selfcode index ausführen.", nil
	}
	return "", nil
}

func preflightApplyAndTest(repoRoot string, diffText string) (string, error) {
	diffText = strings.TrimSpace(diffText)
	if diffText == "" {
		return "", fmt.Errorf("empty diff")
	}
	if _, err := exec.LookPath("git"); err != nil {
		return "", fmt.Errorf("git not found in PATH")
	}
	if _, err := exec.LookPath("go"); err != nil {
		return "", fmt.Errorf("go not found in PATH")
	}

	tmpPatch := filepath.Join(os.TempDir(), fmt.Sprintf("bunny_draft_%d.diff", time.Now().UnixNano()))
	_ = os.WriteFile(tmpPatch, []byte(diffText), 0644)
	defer os.Remove(tmpPatch)

	worktree := filepath.Join(os.TempDir(), fmt.Sprintf("bunny_worktree_%d", time.Now().UnixNano()))
	var log strings.Builder
	log.WriteString("[preflight]\n")
	log.WriteString("repo: " + repoRoot + "\n")
	log.WriteString("worktree: " + worktree + "\n")

	log.WriteString("0) git worktree add --detach\n")
	if out, err := runCmdDir(repoRoot, "git", "worktree", "add", "--detach", worktree, "HEAD"); err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), fmt.Errorf("worktree add failed")
	}
	defer func() {
		_, _ = runCmdDir(repoRoot, "git", "worktree", "remove", "--force", worktree)
		_, _ = runCmdDir(repoRoot, "git", "worktree", "prune")
	}()

	log.WriteString("1) git apply --check\n")
	if out, err := runCmdDir(worktree, "git", "apply", "--check", tmpPatch); err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), fmt.Errorf("git apply --check failed")
	}
	log.WriteString("2) git apply\n")
	if out, err := runCmdDir(worktree, "git", "apply", tmpPatch); err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), fmt.Errorf("git apply failed")
	}
	log.WriteString("3) go test ./...\n")
	testOut, testErr := runCmdDir(worktree, "go", "test", "./...")
	if testErr != nil {
		log.WriteString(testOut + "\n")
		return strings.TrimSpace(log.String()), fmt.Errorf("go test failed")
	}
	if strings.TrimSpace(testOut) != "" {
		log.WriteString(testOut + "\n")
	}
	log.WriteString("OK\n")
	return strings.TrimSpace(log.String()), nil
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

func sanitizeUnifiedDiff(s string) string {
	s = strings.TrimSpace(s)
	// Strip markdown fences if present
	if strings.HasPrefix(s, "```") {
		lines := strings.Split(s, "\n")
		if len(lines) >= 2 {
			// drop first fence line
			lines = lines[1:]
			// drop last fence line if it is a fence
			if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
				lines = lines[:len(lines)-1]
			}
			s = strings.Join(lines, "\n")
		}
	}
	// Drop leading noise before the first diff header
	if i := strings.Index(s, "diff --git "); i > 0 {
		s = s[i:]
	}
	return strings.TrimSpace(s)
}

func firstInvalidHunkLine(diff string) (lineNo int, line string) {
	lines := strings.Split(diff, "\n")
	inHunk := false
	for i, ln := range lines {
		if strings.HasPrefix(ln, "diff --git ") || strings.HasPrefix(ln, "--- ") || strings.HasPrefix(ln, "+++ ") || strings.HasPrefix(ln, "index ") {
			inHunk = false
			continue
		}
		if strings.HasPrefix(ln, "@@") {
			inHunk = true
			continue
		}
		if !inHunk || ln == "" {
			continue
		}
		c := ln[0]
		if c != ' ' && c != '+' && c != '-' && c != '\\' {
			return i + 1, ln
		}
	}
	return 0, ""
}

func validateDraftUnifiedDiff(diff string) (string, error) {
	diff = normalizeUnifiedDiffHunks(strings.TrimSpace(diff))
	if diff == "" {
		return "empty diff", fmt.Errorf("empty diff")
	}
	if n, ln := firstInvalidHunkLine(diff); n > 0 {
		return fmt.Sprintf("invalid hunk line prefix at line %d: %q", n, ln), fmt.Errorf("invalid hunk line prefix")
	}
	repo, err := gitRepoRoot()
	if err != nil {
		// If we can't locate the repo, we can't run git apply --check here.
		return "skip git apply --check (repo root unknown): " + err.Error(), nil
	}
	tmp := filepath.Join(os.TempDir(), fmt.Sprintf("bunny_draft_%d.diff", time.Now().UnixNano()))
	_ = os.WriteFile(tmp, []byte(diff), 0644)
	defer os.Remove(tmp)
	out, err := runCmdDir(repo, "git", "apply", "--check", tmp)
	if err != nil {
		if strings.TrimSpace(out) == "" {
			out = err.Error()
		}
		return "git apply --check failed:\n" + strings.TrimSpace(out), err
	}
	return "git apply --check OK", nil
}

func applyPatchInRepo(id int64, title string, diff string) (string, error) {
	diff = normalizeUnifiedDiffHunks(strings.TrimSpace(diff))
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
	baseBranch, _ := runCmdDir(repo, "git", "rev-parse", "--abbrev-ref", "HEAD")
	baseBranch = strings.TrimSpace(baseBranch)
	if baseBranch == "" {
		baseBranch = "<unknown>"
	}
	branch := fmt.Sprintf("bunny/proposal-%d-%s", id, time.Now().Format("20060102-150405"))
	branch = strings.ReplaceAll(branch, " ", "-")
	if len(branch) > 80 {
		branch = branch[:80]
	}

	var log strings.Builder
	log.WriteString("[code apply]\n")
	log.WriteString("repo: " + repo + "\n")
	log.WriteString("base_branch: " + baseBranch + "\n")
	log.WriteString("new_branch: " + branch + "\n")

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

	log.WriteString("0) git checkout -b\n")
	out, err := runCmdDir(repo, "git", "checkout", "-b", branch)
	if err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), err
	}

	log.WriteString("1) git apply --check\n")
	out, err = runCmdDir(repo, "git", "apply", "--check", tmp)
	if err != nil {
		log.WriteString(out + "\n")
		return strings.TrimSpace(log.String()), err
	}

	log.WriteString("2) git apply\n")
	out, err = runCmdDir(repo, "git", "apply", tmp)
	if err != nil {
		log.WriteString(out + "\n")
		_, _ = runCmdDir(repo, "git", "checkout", baseBranch)
		return strings.TrimSpace(log.String()), err
	}

	log.WriteString("3) go test ./...\n")
	testOut, testErr := runCmdDir(repo, "go", "test", "./...")
	if testErr != nil {
		log.WriteString("go test FAILED:\n" + testOut + "\n")
		rb, _ := runCmdDir(repo, "git", "apply", "-R", tmp)
		log.WriteString("rollback:\n" + rb + "\n")
		_, _ = runCmdDir(repo, "git", "checkout", baseBranch)
		return strings.TrimSpace(log.String()), fmt.Errorf("go test failed; patch rolled back")
	}

	log.WriteString("4) git add -A\n")
	_, _ = runCmdDir(repo, "git", "add", "-A")
	msg := fmt.Sprintf("Apply code_proposal #%d", id)
	if strings.TrimSpace(title) != "" {
		t := strings.TrimSpace(title)
		if len(t) > 64 {
			t = t[:64]
		}
		msg += ": " + t
	}
	log.WriteString("5) git commit\n")
	cout, cerr := runCmdDir(repo, "git", "commit", "-m", msg)
	if cerr != nil {
		if !strings.Contains(strings.ToLower(cout), "nothing to commit") {
			log.WriteString(cout + "\n")
			return strings.TrimSpace(log.String()), cerr
		}
	}
	if strings.TrimSpace(cout) != "" {
		log.WriteString(cout + "\n")
	}

	if strings.TrimSpace(testOut) != "" {
		log.WriteString(testOut + "\n")
	}
	log.WriteString("OK\n")
	log.WriteString("Next: review diff on branch, then merge manually.\n")
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
		kvSet(db, "train_enabled", "1")
		kvSet(db, "ab_enabled", "1")
		return true, "OK. A/B Training ist AN.\n" + renderABStatus(db, eg)
	case "off":
		kvSet(db, "train_enabled", "0")
		kvSet(db, "ab_enabled", "0")
		return true, "OK. A/B Training ist AUS."
	case "status":
		return true, renderABStatus(db, eg)
	case "explain":
		if len(parts) < 3 {
			return true, "Use: /ab explain on|off"
		}
		v2 := strings.ToLower(strings.TrimSpace(parts[2]))
		switch v2 {
		case "on", "1", "true":
			kvSet(db, "train_explain", "1")
			return true, "OK. explain=ON\n" + renderABStatus(db, eg)
		case "off", "0", "false":
			kvSet(db, "train_explain", "0")
			return true, "OK. explain=OFF"
		default:
			return true, "Use: /ab explain on|off"
		}
	case "set":
		if len(parts) < 4 {
			return true, "Use: /ab set b_model|b_style|mutant_strength|pool <value>"
		}
		k := strings.ToLower(strings.TrimSpace(parts[2]))
		v := strings.TrimSpace(strings.TrimPrefix(line, parts[0]+" "+parts[1]+" "+parts[2]))
		v = strings.TrimSpace(v)
		if v == "" {
			return true, "Value leer."
		}
		switch k {
		case "b_model":
			// Optional override. If empty, Bunny auto-picks a mutant model or uses the same model.
			kvSet(db, "train_mutant_model", v)
			kvSet(db, "ab_model_b", v) // legacy
			return true, "OK. b_model gesetzt.\n" + renderABStatus(db, eg)
		case "b_style":
			// Persistent mutant overlay (phenotype)
			kvSet(db, "train_mutant_prompt", v)
			kvSet(db, "ab_style_b", v)
			return true, "OK. b_style gesetzt.\n" + renderABStatus(db, eg)
		case "mutant_strength":
			kvSet(db, "train_mutant_strength", v)
			return true, "OK. mutant_strength gesetzt.\n" + renderABStatus(db, eg)
		case "pool":
			kvSet(db, "train_model_pool", v)
			return true, "OK. pool gesetzt.\n" + renderABStatus(db, eg)
		default:
			return true, "Use: /ab set b_model|b_style|mutant_strength|pool <value>"
		}
	default:
		return true, "Use: /ab on|off|status|set|explain ..."
	}
}

func handlePickCommand(db *sql.DB, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/pick") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) < 3 {
		return true, "Use: /pick <id> A|B|none"
	}
	id := parseID(parts[1])
	choiceRaw := strings.TrimSpace(parts[2])
	choice := strings.ToUpper(choiceRaw)
	if choice == "A" || choice == "B" || strings.EqualFold(choice, "none") {
		if strings.EqualFold(choice, "none") {
			choice = "NONE"
		}
		// Prefer train_trials (online learning)
		if tt, ok := brain.GetTrainTrialFull(db, id); ok {
			if tt.Chosen != "" {
				switch strings.ToUpper(tt.Chosen) {
				case "A":
					return true, tt.AText
				case "B":
					return true, tt.BText
				default:
					return true, "OK. (none)"
				}
			}
			if choice == "NONE" {
				_ = brain.ChooseTrainTrial(db, id, "NONE")
				kvSet(db, "speech_overlay", "")
				kvSet(db, "speaker_model_override", "")
				return true, "OK. (none)"
			}
			_ = brain.ChooseTrainTrial(db, id, choice)

			// Capture before-stats for explain mode.
			ctxKey := tt.CtxKey
			aAct := tt.AAction
			bAct := tt.BAction
			aSty := tt.AStyle
			bSty := tt.BStyle
			aA0, aB0 := getPolicyAlphaBeta(db, ctxKey, aAct)
			bA0, bB0 := getPolicyAlphaBeta(db, ctxKey, bAct)
			psA0 := getPrefValue(db, "style:"+aSty)
			psB0 := getPrefValue(db, "style:"+bSty)
			ptA0 := getPrefValue(db, "strat:"+aAct)
			ptB0 := getPrefValue(db, "strat:"+bAct)

			brain.ApplyTrainChoice(db, id, choice)

			aA1, aB1 := getPolicyAlphaBeta(db, ctxKey, aAct)
			bA1, bB1 := getPolicyAlphaBeta(db, ctxKey, bAct)
			psA1 := getPrefValue(db, "style:"+aSty)
			psB1 := getPrefValue(db, "style:"+bSty)
			ptA1 := getPrefValue(db, "strat:"+aAct)
			ptB1 := getPrefValue(db, "strat:"+bAct)

			learned := ""
			if trainExplainEnabled(db) {
				learned = "LEARNED (ctx=" + ctxKey + ", choice=" + choice + ")\n" +
					"- policy[" + aAct + "] α/β: " + fmtAB(aA0, aB0) + " → " + fmtAB(aA1, aB1) + "\n" +
					"- policy[" + bAct + "] α/β: " + fmtAB(bA0, bB0) + " → " + fmtAB(bA1, bB1) + "\n" +
					"- pref[style:" + aSty + "]: " + fmtF(psA0) + " → " + fmtF(psA1) + "\n" +
					"- pref[style:" + bSty + "]: " + fmtF(psB0) + " → " + fmtF(psB1) + "\n" +
					"- pref[strat:" + aAct + "]: " + fmtF(ptA0) + " → " + fmtF(ptA1) + "\n" +
					"- pref[strat:" + bAct + "]: " + fmtF(ptB0) + " → " + fmtF(ptB1) + "\n"
			}

			// Apply phenotype immediately based on stored note.
			if choice == "B" {
				// Note JSON is optional; fall back to B style text.
				overlay := strings.TrimSpace(tt.BStyle)
				m := ""
				if strings.TrimSpace(tt.Note) != "" {
					var meta struct {
						AModel  string `json:"a_model"`
						BModel  string `json:"b_model"`
						BPrompt string `json:"b_prompt"`
					}
					if json.Unmarshal([]byte(tt.Note), &meta) == nil {
						if strings.TrimSpace(meta.BPrompt) != "" {
							overlay = strings.TrimSpace(meta.BPrompt)
						}
						m = strings.TrimSpace(meta.BModel)
					}
				}
				if overlay != "" {
					kvSet(db, "speech_overlay", overlay)
				}
				if m != "" {
					kvSet(db, "speaker_model_override", m)
				}
				if learned != "" {
					learned += "- phenotype: speech_overlay=" + firstLine(overlay) + "; speaker_model_override=" + pickNonEmpty(m, "<same>") + "\n"
				}
				if learned != "" {
					return true, strings.TrimSpace(learned + "\n\n" + tt.BText)
				}
				return true, tt.BText
			}
			// choice A
			kvSet(db, "speech_overlay", "")
			kvSet(db, "speaker_model_override", "")
			if learned != "" {
				learned += "- phenotype: cleared speech_overlay + model_override\n"
			}
			if learned != "" {
				return true, strings.TrimSpace(learned + "\n\n" + tt.AText)
			}
			return true, tt.AText
		}
	}

	// Legacy fallback: AB trials
	choice2 := strings.ToLower(strings.TrimSpace(choiceRaw))
	t, ok := brain.GetABTrial(db, id)
	if !ok {
		return true, "ID nicht gefunden."
	}
	if t.Status == "chosen" {
		switch t.Choice {
		case "a":
			return true, t.AText
		case "b":
			return true, t.BText
		default:
			return true, "OK. (none)"
		}
	}
	if err := brain.ChooseABTrial(db, id, choice2); err != nil {
		return true, "ERR: " + err.Error()
	}
	switch choice2 {
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

func trainEnabled(db *sql.DB) bool {
	if strings.TrimSpace(kvGet(db, "train_enabled")) == "1" {
		return true
	}
	return abEnabled(db)
}

func trainExplainEnabled(db *sql.DB) bool {
	// default ON
	v := strings.TrimSpace(kvGet(db, "train_explain"))
	if v == "0" {
		return false
	}
	return true
}

func getPolicyAlphaBeta(db *sql.DB, ctx, action string) (float64, float64) {
	if db == nil || strings.TrimSpace(ctx) == "" || strings.TrimSpace(action) == "" {
		return 1.0, 1.0
	}
	a, b := 1.0, 1.0
	_ = db.QueryRow(`SELECT alpha,beta FROM policy_stats WHERE context_key=? AND action=?`, ctx, action).Scan(&a, &b)
	if a == 0 && b == 0 {
		a, b = 1.0, 1.0
	}
	if a < 0.1 {
		a = 0.1
	}
	if b < 0.1 {
		b = 0.1
	}
	return a, b
}

func getPrefValue(db *sql.DB, key string) float64 {
	if db == nil || strings.TrimSpace(key) == "" {
		return 0
	}
	var v sql.NullFloat64
	_ = db.QueryRow(`SELECT value FROM preferences WHERE key=?`, key).Scan(&v)
	if !v.Valid {
		return 0
	}
	return v.Float64
}

func fmtAB(a, b float64) string {
	return fmt.Sprintf("%.2f/%.2f", a, b)
}

func fmtF(x float64) string {
	return fmt.Sprintf("%.2f", x)
}

func renderABStatus(db *sql.DB, eg *epi.Epigenome) string {
	champ := strings.TrimSpace(kvGet(db, "speaker_model_override"))
	if champ == "" {
		champ = eg.ModelFor("speaker", "llama3.1:8b")
	}
	mutModel := strings.TrimSpace(kvGet(db, "train_mutant_model"))
	mutPrompt := strings.TrimSpace(kvGet(db, "train_mutant_prompt"))
	if mutPrompt == "" {
		mutPrompt = "STYLE: empathisch-direkt. Kurz, warm, direkt. Keine unnötigen Rückfragen. 2–5 Sätze."
	}
	mutStr := strings.TrimSpace(kvGet(db, "train_mutant_strength"))
	if mutStr == "" {
		mutStr = "0.20"
	}
	pool := strings.TrimSpace(kvGet(db, "train_model_pool"))

	en := "OFF"
	if trainEnabled(db) {
		en = "ON"
	}
	return "A/B Training: " + en + "\n" +
		"Champion model: " + champ + "\n" +
		"Mutant model: " + pickNonEmpty(mutModel, "<auto>") + "\n" +
		"Mutant strength: " + mutStr + "\n" +
		"Mutant prompt: " + firstLine(mutPrompt) + "\n" +
		"Model pool: " + pickNonEmpty(pool, "<auto>") + "\n" +
		"Explain: " + pickNonEmpty(kvGet(db, "train_explain"), "1") + " (1=on,0=off)\n\n" +
		"Tipps: /ab set b_style <prompt> | /ab set b_model <model> | /ab set pool <csv> | /ab explain on|off | /ab off"
}

func firstLine(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[:i]
	}
	if len(s) > 90 {
		s = s[:90] + "..."
	}
	return s
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

func runTrainTrial(db *sql.DB, epiPath string, oc *ollama.Client, modelSpeaker, modelStance string, body *BodyState, aff *brain.AffectState, ws *brain.Workspace, tr *brain.Traits, dr *brain.Drives, eg *epi.Epigenome, userText string) (string, bool) {
	// Champion model: current override or configured speaker.
	aModel := strings.TrimSpace(kvGet(db, "speaker_model_override"))
	if aModel == "" {
		aModel = eg.ModelFor("speaker", modelSpeaker)
	}

	// Mutant model: optional override; otherwise auto-pick or fall back to champion.
	bModel := strings.TrimSpace(kvGet(db, "train_mutant_model"))
	if bModel == "" {
		bModel = autoPickMutantModel(oc, strings.TrimSpace(kvGet(db, "train_model_pool")), aModel)
	}
	if bModel == "" {
		bModel = aModel
	}

	mutPrompt := strings.TrimSpace(kvGet(db, "train_mutant_prompt"))
	if mutPrompt == "" {
		mutPrompt = "STYLE: empathisch-direkt. Kurz, warm, direkt. Keine unnötigen Rückfragen. 2–5 Sätze."
	}
	mutStrength := 0.20
	if v := strings.TrimSpace(kvGet(db, "train_mutant_strength")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			mutStrength = f
		}
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
	if wsA != nil {
		wsA.TrainingDryRun = true
	}
	if wsB != nil {
		wsB.TrainingDryRun = true
	}

	aOut, aAct, aSty, ctxKey, topic, intentMode := ExecuteTurnWithMeta(db, epiPath, oc, aModel, modelStance, &bodyA, affA, wsA, tr, &drA, eg, userText, nil)
	mut := &MutantOverlay{Strength: mutStrength, Prompt: mutPrompt, Model: bModel}
	bOut, bAct, bSty, _, _, _ := ExecuteTurnWithMeta(db, epiPath, oc, aModel, modelStance, &bodyB, affB, wsB, tr, &drB, eg, userText, mut)
	aOut = strings.TrimSpace(aOut)
	bOut = strings.TrimSpace(bOut)
	if aOut == "" || bOut == "" {
		return "Train-Trial konnte nicht erzeugt werden (leere Kandidaten).", false
	}

	userMsgID := int64(0)
	if ws != nil {
		userMsgID = ws.LastUserMsgID
	}
	id, err := brain.InsertTrainTrial(db, userMsgID, topic, intentMode, ctxKey, aAct, aSty, aOut, bAct, bSty, bOut)
	if err != nil {
		return "ERR: " + err.Error(), true
	}
	meta := map[string]any{
		"a_model":      aModel,
		"b_model":      bModel,
		"a_action":     aAct,
		"b_action":     bAct,
		"a_style":      aSty,
		"b_style":      bSty,
		"b_prompt":     mutPrompt,
		"mut_strength": mutStrength,
		"ctx_key":      ctxKey,
		"topic":        topic,
		"intent":       intentMode,
		"weights_diff": !strings.EqualFold(strings.TrimSpace(aModel), strings.TrimSpace(bModel)),
	}
	metaJSON, _ := json.Marshal(meta)
	_ = brain.UpdateTrainTrialNote(db, id, string(metaJSON))

	var b strings.Builder
	b.WriteString("TRAIN#" + strconv.FormatInt(id, 10) + "\n")
	b.WriteString("A (" + aModel + ", action=" + aAct + ", style=" + aSty + "):\n" + aOut + "\n\n")
	b.WriteString("B (" + bModel + ", action=" + bAct + ", style=" + bSty + "):\n" + bOut + "\n\n")

	// Variation vector (what exactly differs between A and B)
	wDiff := "SAME"
	if !strings.EqualFold(strings.TrimSpace(aModel), strings.TrimSpace(bModel)) {
		wDiff = "DIFF (different model)"
	}
	over := firstLine(mutPrompt)
	if over == "" {
		over = "<none>"
	}
	varV := "VARIATION\n" +
		"- ctx: " + ctxKey + "\n" +
		"- topic: " + pickNonEmpty(topic, "<none>") + "\n" +
		"- intent: " + intentMode + "\n" +
		"- Δmodel: " + aModel + " → " + bModel + "\n" +
		"- Δweights: " + wDiff + "\n" +
		"- Δaction: " + aAct + " → " + bAct + "\n" +
		"- Δstyle: " + aSty + " → " + bSty + "\n" +
		"- Δoverlay(B): " + over + " (strength=" + fmtF(mutStrength) + ")\n" +
		"- Δepigenome (trial): none; learning happens on /pick (policy_stats, preferences, kv_state)\n"
	b.WriteString(varV + "\n")

	b.WriteString("Wähle: /pick " + strconv.FormatInt(id, 10) + " A|B|none")
	return strings.TrimSpace(b.String()), true
}

func autoPickMutantModel(oc *ollama.Client, poolCSV string, fallback string) string {
	poolCSV = strings.TrimSpace(poolCSV)
	if poolCSV != "" {
		parts := strings.Split(poolCSV, ",")
		for _, p := range parts {
			m := strings.TrimSpace(p)
			if m != "" && m != fallback {
				return m
			}
		}
	}
	if oc == nil {
		return ""
	}
	models, err := oc.ListModels()
	if err != nil || len(models) == 0 {
		return ""
	}
	// Prefer bunny-* or *lora* or *adapter*.
	cands := []string{}
	for m := range models {
		lm := strings.ToLower(strings.TrimSpace(m))
		if lm == "" || lm == strings.ToLower(fallback) {
			continue
		}
		if strings.HasPrefix(lm, "bunny-") || strings.Contains(lm, "lora") || strings.Contains(lm, "adapter") {
			cands = append(cands, m)
		}
	}
	if len(cands) > 0 {
		idx := int(time.Now().UnixNano() % int64(len(cands)))
		if idx < 0 {
			idx = -idx
		}
		return cands[idx]
	}
	// If no LoRA/adapter models exist, still vary by picking ANY other installed model.
	for m := range models {
		lm := strings.ToLower(strings.TrimSpace(m))
		if lm == "" || lm == strings.ToLower(fallback) {
			continue
		}
		cands = append(cands, m)
	}
	if len(cands) == 0 {
		return ""
	}
	idx := int(time.Now().UnixNano() % int64(len(cands)))
	if idx < 0 {
		idx = -idx
	}
	return cands[idx]
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

func isStopProposalSpam(s string) bool {
	t := strings.ToLower(strings.TrimSpace(s))
	if t == "" {
		return false
	}
	neg := strings.Contains(t, "nicht")
	spam := strings.Contains(t, "dauer") || strings.Contains(t, "ständig") || strings.Contains(t, "permanent") || strings.Contains(t, "spam")
	idea := strings.Contains(t, "idee") || strings.Contains(t, "vorschlag") || strings.Contains(t, "verbesser") || strings.Contains(t, "thought_proposal")
	return neg && spam && idea
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

func handleEpiCommands(db *sql.DB, epiPath string, eg *epi.Epigenome, userText string) (bool, string) {
	line := strings.TrimSpace(userText)
	if !strings.HasPrefix(line, "/epi") {
		return false, ""
	}
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return true, "Use: /epi list | /epi show <id> | /epi apply <id> | /epi reject <id> | /epi dump"
	}
	sub := strings.ToLower(strings.TrimSpace(parts[1]))
	switch sub {
	case "list":
		return true, brain.RenderEpigenomeProposalList(db, 20)
	case "show":
		if len(parts) < 3 {
			return true, "Use: /epi show <id>"
		}
		id := parseID(parts[2])
		if id <= 0 {
			return true, "Bad id."
		}
		return true, brain.RenderEpigenomeProposal(db, id)
	case "reject":
		if len(parts) < 3 {
			return true, "Use: /epi reject <id>"
		}
		id := parseID(parts[2])
		if id <= 0 {
			return true, "Bad id."
		}
		brain.MarkEpigenomeProposal(db, id, "rejected")
		return true, "OK. (rejected)"
	case "dump":
		b, err := os.ReadFile(epiPath)
		if err != nil {
			return true, "ERR: " + err.Error()
		}
		return true, string(b)
	case "apply":
		if len(parts) < 3 {
			return true, "Use: /epi apply <id>"
		}
		id := parseID(parts[2])
		if id <= 0 {
			return true, "Bad id."
		}
		row, ok := brain.GetEpigenomeProposal(db, id)
		if !ok {
			return true, "Nicht gefunden."
		}
		if strings.TrimSpace(row.Status) != "proposed" {
			return true, "Nicht offen (status=" + row.Status + ")"
		}
		cur, err := epi.LoadOrInit(epiPath)
		if err != nil {
			return true, "ERR load epigenome: " + err.Error()
		}
		next, err := cur.ApplyMergePatch([]byte(row.PatchJSON))
		if err != nil {
			return true, "ERR patch: " + err.Error()
		}
		if err := next.Save(epiPath); err != nil {
			return true, "ERR save: " + err.Error()
		}
		if eg != nil {
			*eg = *next
		}
		brain.MarkEpigenomeProposal(db, id, "applied")
		return true, "OK. Epigenome patch applied: #" + strconv.FormatInt(id, 10)
	default:
		return true, "Use: /epi list | /epi show <id> | /epi apply <id> | /epi reject <id> | /epi dump"
	}
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
