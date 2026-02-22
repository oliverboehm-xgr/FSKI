package brain

import (
	"database/sql"
	"strconv"
	"strings"
	"time"
)

type ProposalRow struct {
	ID        int64
	CreatedAt string
	Title     string
	Status    string
}

func CountPendingProposals(db *sql.DB) (schema int, code int) {
	if db == nil {
		return 0, 0
	}
	_ = db.QueryRow(`SELECT COUNT(*) FROM schema_proposals WHERE status='proposed'`).Scan(&schema)
	_ = db.QueryRow(`SELECT COUNT(*) FROM code_proposals WHERE status='proposed'`).Scan(&code)
	return
}

func InsertSchemaProposal(db *sql.DB, title, sqlText, notes string) (int64, error) {
	if db == nil {
		return 0, nil
	}
	title = strings.TrimSpace(title)
	sqlText = strings.TrimSpace(sqlText)
	if title == "" {
		title = "schema_change"
	}
	if sqlText == "" {
		return 0, nil
	}
	now := time.Now().Format(time.RFC3339)
	res, err := db.Exec(`INSERT INTO schema_proposals(created_at,title,sql,status,notes) VALUES(?,?,?,?,?)`,
		now, title, sqlText, "proposed", notes)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func InsertCodeProposal(db *sql.DB, title, diffText, notes string) (int64, error) {
	if db == nil {
		return 0, nil
	}
	title = strings.TrimSpace(title)
	diffText = strings.TrimSpace(diffText)
	if title == "" {
		title = "code_change"
	}
	if diffText == "" {
		return 0, nil
	}
	now := time.Now().Format(time.RFC3339)
	res, err := db.Exec(`INSERT INTO code_proposals(created_at,title,diff,status,notes) VALUES(?,?,?,?,?)`,
		now, title, diffText, "proposed", notes)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func ListSchemaProposals(db *sql.DB, status string, limit int) ([]ProposalRow, error) {
	if db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 20
	}
	q := `SELECT id, created_at, title, status FROM schema_proposals`
	var args []any
	if status != "" {
		q += ` WHERE status=?`
		args = append(args, status)
	}
	q += ` ORDER BY id DESC LIMIT ?`
	args = append(args, limit)
	rows, err := db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ProposalRow
	for rows.Next() {
		var r ProposalRow
		_ = rows.Scan(&r.ID, &r.CreatedAt, &r.Title, &r.Status)
		out = append(out, r)
	}
	return out, nil
}

func ListCodeProposals(db *sql.DB, status string, limit int) ([]ProposalRow, error) {
	if db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 20
	}
	q := `SELECT id, created_at, title, status FROM code_proposals`
	var args []any
	if status != "" {
		q += ` WHERE status=?`
		args = append(args, status)
	}
	q += ` ORDER BY id DESC LIMIT ?`
	args = append(args, limit)
	rows, err := db.Query(q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ProposalRow
	for rows.Next() {
		var r ProposalRow
		_ = rows.Scan(&r.ID, &r.CreatedAt, &r.Title, &r.Status)
		out = append(out, r)
	}
	return out, nil
}

func GetSchemaProposal(db *sql.DB, id int64) (title, sqlText, status string, ok bool) {
	if db == nil || id <= 0 {
		return "", "", "", false
	}
	_ = db.QueryRow(`SELECT title, sql, status FROM schema_proposals WHERE id=?`, id).Scan(&title, &sqlText, &status)
	title = strings.TrimSpace(title)
	sqlText = strings.TrimSpace(sqlText)
	status = strings.TrimSpace(status)
	return title, sqlText, status, sqlText != ""
}

func GetCodeProposal(db *sql.DB, id int64) (title, diffText, status string, ok bool) {
	if db == nil || id <= 0 {
		return "", "", "", false
	}
	_ = db.QueryRow(`SELECT title, diff, status FROM code_proposals WHERE id=?`, id).Scan(&title, &diffText, &status)
	title = strings.TrimSpace(title)
	diffText = strings.TrimSpace(diffText)
	status = strings.TrimSpace(status)
	return title, diffText, status, diffText != ""
}

func GetCodeProposalFull(db *sql.DB, id int64) (title, diffText, status, notes string, ok bool) {
	if db == nil || id <= 0 {
		return "", "", "", "", false
	}
	_ = db.QueryRow(`SELECT title, diff, status, notes FROM code_proposals WHERE id=?`, id).Scan(&title, &diffText, &status, &notes)
	title = strings.TrimSpace(title)
	diffText = strings.TrimSpace(diffText)
	status = strings.TrimSpace(status)
	notes = strings.TrimSpace(notes)
	return title, diffText, status, notes, diffText != "" || notes != ""
}

func UpdateCodeProposal(db *sql.DB, id int64, diffText string, status string) {
	if db == nil || id <= 0 {
		return
	}
	diffText = strings.TrimSpace(diffText)
	status = strings.TrimSpace(status)
	if status == "" {
		status = "proposed"
	}
	_, _ = db.Exec(`UPDATE code_proposals SET diff=?, status=? WHERE id=?`, diffText, status, id)
}

func RenderCodeProposalList(db *sql.DB, limit int) string {
	items, err := ListCodeProposals(db, "", limit)
	if err != nil || len(items) == 0 {
		return "Keine code_proposals gefunden."
	}
	var b strings.Builder
	b.WriteString("code_proposals (neueste zuerst):\n")
	for _, it := range items {
		b.WriteString("- #" + strconv.FormatInt(it.ID, 10) + " [" + it.Status + "] " + strings.TrimSpace(it.Title) + "\n")
	}
	b.WriteString("\nNutzen:\n- /code show <id>\n- /code draft <id>\n- /code apply <id>\n")
	return strings.TrimSpace(b.String())
}

func RenderCodeProposal(db *sql.DB, id int64) string {
	title, diffText, status, notes, ok := GetCodeProposalFull(db, id)
	if !ok {
		return "Nicht gefunden."
	}
	var b strings.Builder
	b.WriteString("code_proposal #" + strconv.FormatInt(id, 10) + "\n")
	b.WriteString("status: " + status + "\n")
	b.WriteString("title: " + title + "\n")
	if notes != "" {
		b.WriteString("\nnotes:\n" + notes + "\n")
	}
	if diffText != "" {
		b.WriteString("\ndiff:\n" + diffText + "\n")
	}
	b.WriteString("\nWeiter:\n- /code draft " + strconv.FormatInt(id, 10) + "\n- /code apply " + strconv.FormatInt(id, 10))
	return strings.TrimSpace(b.String())
}

func MarkSchemaProposal(db *sql.DB, id int64, status string) {
	if db == nil || id <= 0 {
		return
	}
	status = strings.TrimSpace(status)
	if status == "" {
		return
	}
	_, _ = db.Exec(`UPDATE schema_proposals SET status=? WHERE id=?`, status, id)
}

func MarkCodeProposal(db *sql.DB, id int64, status string) {
	if db == nil || id <= 0 {
		return
	}
	status = strings.TrimSpace(status)
	if status == "" {
		return
	}
	_, _ = db.Exec(`UPDATE code_proposals SET status=? WHERE id=?`, status, id)
}
