package brain

import (
	"database/sql"
	"fmt"
	"strconv"
	"strings"
	"time"
)

type ThoughtProposal struct {
	ID        int64
	CreatedAt string
	Kind      string
	Title     string
	Payload   string
	Status    string
	Note      string
}

func ListThoughtProposals(db *sql.DB, status string, limit int) ([]ThoughtProposal, error) {
	if db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 10
	}
	q := `SELECT id, created_at, kind, title, payload, status, note FROM thought_proposals`
	var args []any
	if strings.TrimSpace(status) != "" {
		q += ` WHERE status=?`
		args = append(args, strings.TrimSpace(status))
	}
	q += ` ORDER BY id DESC LIMIT ?`
	args = append(args, limit)

	rows, err := db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []ThoughtProposal
	for rows.Next() {
		var r ThoughtProposal
		_ = rows.Scan(&r.ID, &r.CreatedAt, &r.Kind, &r.Title, &r.Payload, &r.Status, &r.Note)
		r.Kind = strings.TrimSpace(r.Kind)
		r.Title = strings.TrimSpace(r.Title)
		r.Payload = strings.TrimSpace(r.Payload)
		r.Status = strings.TrimSpace(r.Status)
		r.Note = strings.TrimSpace(r.Note)
		out = append(out, r)
	}
	return out, nil
}

func GetThoughtProposal(db *sql.DB, id int64) (ThoughtProposal, bool) {
	if db == nil || id <= 0 {
		return ThoughtProposal{}, false
	}
	var r ThoughtProposal
	_ = db.QueryRow(`SELECT id, created_at, kind, title, payload, status, note FROM thought_proposals WHERE id=?`, id).
		Scan(&r.ID, &r.CreatedAt, &r.Kind, &r.Title, &r.Payload, &r.Status, &r.Note)
	r.Kind = strings.TrimSpace(r.Kind)
	r.Title = strings.TrimSpace(r.Title)
	r.Payload = strings.TrimSpace(r.Payload)
	r.Status = strings.TrimSpace(r.Status)
	r.Note = strings.TrimSpace(r.Note)
	return r, r.ID > 0
}

func MarkThoughtProposal(db *sql.DB, id int64, status string) {
	if db == nil || id <= 0 {
		return
	}
	status = strings.TrimSpace(status)
	if status == "" {
		return
	}
	_, _ = db.Exec(`UPDATE thought_proposals SET status=? WHERE id=?`, status, id)
}

func CountThoughtProposals(db *sql.DB, status string) int {
	if db == nil {
		return 0
	}
	q := `SELECT COUNT(*) FROM thought_proposals`
	var args []any
	if strings.TrimSpace(status) != "" {
		q += ` WHERE status=?`
		args = append(args, strings.TrimSpace(status))
	}
	var n int
	_ = db.QueryRow(q, args...).Scan(&n)
	return n
}

func RenderThoughtProposalList(db *sql.DB, limit int) string {
	items, err := ListThoughtProposals(db, "proposed", limit)
	if err != nil || len(items) == 0 {
		return "Ich habe gerade keine offenen thought_proposals."
	}
	var b strings.Builder
	b.WriteString("Offene thought_proposals:\n")
	for _, it := range items {
		b.WriteString(fmt.Sprintf("- #%d [%s] %s\n", it.ID, safe(it.Kind), safe(it.Title)))
	}
	b.WriteString("\nNutzen:\n")
	b.WriteString("- /thought show <id>\n")
	b.WriteString("- /thought materialize <id|all>\n")
	return b.String()
}

func RenderThoughtProposal(db *sql.DB, id int64) string {
	it, ok := GetThoughtProposal(db, id)
	if !ok {
		return "Nicht gefunden."
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("thought_proposal #%d\n", it.ID))
	b.WriteString("created_at: " + it.CreatedAt + "\n")
	b.WriteString("kind: " + safe(it.Kind) + "\n")
	b.WriteString("title: " + safe(it.Title) + "\n")
	if it.Note != "" {
		b.WriteString("note: " + it.Note + "\n")
	}
	b.WriteString("\npayload:\n")
	b.WriteString(it.Payload)
	b.WriteString("\n\nWeiter:\n- /thought materialize " + strconv.FormatInt(it.ID, 10))
	return b.String()
}

// MaterializeThoughtProposal converts a thought_idea into a concrete schema/code proposal placeholder.
// v0: We do NOT auto-generate diffs/SQL. We create a concrete proposal record + keep the payload as notes,
// so the pipeline is reviewable and can later be enhanced by CodeIndex/LLM.
func MaterializeThoughtProposal(db *sql.DB, id int64) (string, bool) {
	it, ok := GetThoughtProposal(db, id)
	if !ok || it.Status != "proposed" {
		return "Kein offenes thought_proposal mit dieser ID.", false
	}
	now := time.Now().Format(time.RFC3339)
	notes := strings.TrimSpace(it.Payload)
	if it.Note != "" {
		notes = strings.TrimSpace(notes + "\n\nNOTE: " + it.Note)
	}
	switch strings.ToLower(it.Kind) {
	case "schema":
		sqlText := "-- TODO: fill SQL for: " + it.Title + "\n-- From thought_proposals#" + strconv.FormatInt(id, 10) + "\n"
		pid, err := InsertSchemaProposal(db, it.Title, sqlText, notes)
		if err == nil && pid > 0 {
			MarkThoughtProposal(db, id, "materialized")
			_ = now
			return "OK. Als schema_proposal gespeichert: #" + strconv.FormatInt(pid, 10) + " (aus thought_proposal #" + strconv.FormatInt(id, 10) + ")\nNutze: /schema show " + strconv.FormatInt(pid, 10), true
		}
	case "code":
		diffText := "# TODO: implement code change for: " + it.Title + "\n# From thought_proposals#" + strconv.FormatInt(id, 10) + "\n"
		pid, err := InsertCodeProposal(db, it.Title, diffText, notes)
		if err == nil && pid > 0 {
			MarkThoughtProposal(db, id, "materialized")
			_ = now
			return "OK. Als code_proposal gespeichert: #" + strconv.FormatInt(pid, 10) + " (aus thought_proposal #" + strconv.FormatInt(id, 10) + ")\nNutze: /code show " + strconv.FormatInt(pid, 10), true
		}
	case "epigenetic":
		// Keep as thought unless you want auto-apply. We materialize to schema_proposals as placeholder.
		diffText := "# epigenetic idea: " + it.Title + "\n" + it.Payload + "\n"
		pid, err := InsertCodeProposal(db, "epigenetic:"+it.Title, diffText, notes)
		if err == nil && pid > 0 {
			MarkThoughtProposal(db, id, "materialized")
			return "OK. Als code_proposal gespeichert: #" + strconv.FormatInt(pid, 10) + " (epigenetic)\nNutze: /code show " + strconv.FormatInt(pid, 10), true
		}
	default:
		// unknown -> keep but mark reviewed?
	}
	return "Konnte nicht materialisieren (DB error oder unknown kind).", false
}

func MaterializeAllThoughtProposals(db *sql.DB, limit int) string {
	items, err := ListThoughtProposals(db, "proposed", limit)
	if err != nil || len(items) == 0 {
		return "Keine offenen thought_proposals."
	}
	okN := 0
	var b strings.Builder
	for _, it := range items {
		msg, ok := MaterializeThoughtProposal(db, it.ID)
		if ok {
			okN++
		} else {
			b.WriteString("FAIL #" + strconv.FormatInt(it.ID, 10) + ": " + msg + "\n")
		}
	}
	if okN > 0 {
		b.WriteString(fmt.Sprintf("Materialisiert: %d\n", okN))
	}
	return strings.TrimSpace(b.String())
}

func safe(s string) string { return strings.TrimSpace(s) }
