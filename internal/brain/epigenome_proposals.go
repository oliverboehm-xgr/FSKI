package brain

import (
	"database/sql"
	"strconv"
	"strings"
	"time"
)

type EpigenomeProposal struct {
	ID        int64
	CreatedAt string
	Title     string
	PatchJSON string
	Status    string
	Notes     string
}

func InsertEpigenomeProposal(db *sql.DB, title, patchJSON, notes string) (int64, error) {
	if db == nil {
		return 0, nil
	}
	title = strings.TrimSpace(title)
	patchJSON = strings.TrimSpace(patchJSON)
	notes = strings.TrimSpace(notes)
	if title == "" {
		title = "epigenome_change"
	}
	if patchJSON == "" {
		return 0, nil
	}
	now := time.Now().Format(time.RFC3339)
	res, err := db.Exec(`INSERT INTO epigenome_proposals(created_at,title,patch_json,status,notes) VALUES(?,?,?,?,?)`,
		now, title, patchJSON, "proposed", notes)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func ListEpigenomeProposals(db *sql.DB, status string, limit int) ([]EpigenomeProposal, error) {
	if db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 20
	}
	q := `SELECT id, created_at, title, patch_json, status, notes FROM epigenome_proposals`
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
	var out []EpigenomeProposal
	for rows.Next() {
		var r EpigenomeProposal
		_ = rows.Scan(&r.ID, &r.CreatedAt, &r.Title, &r.PatchJSON, &r.Status, &r.Notes)
		r.Title = strings.TrimSpace(r.Title)
		r.PatchJSON = strings.TrimSpace(r.PatchJSON)
		r.Status = strings.TrimSpace(r.Status)
		r.Notes = strings.TrimSpace(r.Notes)
		out = append(out, r)
	}
	return out, nil
}

func GetEpigenomeProposal(db *sql.DB, id int64) (EpigenomeProposal, bool) {
	if db == nil || id <= 0 {
		return EpigenomeProposal{}, false
	}
	var r EpigenomeProposal
	_ = db.QueryRow(`SELECT id, created_at, title, patch_json, status, notes FROM epigenome_proposals WHERE id=?`, id).
		Scan(&r.ID, &r.CreatedAt, &r.Title, &r.PatchJSON, &r.Status, &r.Notes)
	r.Title = strings.TrimSpace(r.Title)
	r.PatchJSON = strings.TrimSpace(r.PatchJSON)
	r.Status = strings.TrimSpace(r.Status)
	r.Notes = strings.TrimSpace(r.Notes)
	return r, r.ID > 0
}

func MarkEpigenomeProposal(db *sql.DB, id int64, status string) {
	if db == nil || id <= 0 {
		return
	}
	status = strings.TrimSpace(status)
	if status == "" {
		return
	}
	_, _ = db.Exec(`UPDATE epigenome_proposals SET status=? WHERE id=?`, status, id)
}

func RenderEpigenomeProposalList(db *sql.DB, limit int) string {
	items, err := ListEpigenomeProposals(db, "", limit)
	if err != nil || len(items) == 0 {
		return "Keine epigenome_proposals gefunden."
	}
	var b strings.Builder
	b.WriteString("epigenome_proposals (neueste zuerst):\n")
	for _, it := range items {
		b.WriteString("- #" + strconv.FormatInt(it.ID, 10) + " [" + it.Status + "] " + strings.TrimSpace(it.Title) + "\n")
	}
	b.WriteString("\nNutzen:\n- /epi show <id>\n- /epi apply <id>\n- /epi reject <id>\n")
	return strings.TrimSpace(b.String())
}

func RenderEpigenomeProposal(db *sql.DB, id int64) string {
	it, ok := GetEpigenomeProposal(db, id)
	if !ok {
		return "Nicht gefunden."
	}
	var b strings.Builder
	b.WriteString("epigenome_proposal #" + strconv.FormatInt(id, 10) + "\n")
	b.WriteString("status: " + it.Status + "\n")
	b.WriteString("title: " + it.Title + "\n")
	if it.Notes != "" {
		b.WriteString("\nnotes:\n" + it.Notes + "\n")
	}
	b.WriteString("\npatch_json (merge patch):\n" + it.PatchJSON + "\n")
	b.WriteString("\nWeiter:\n- /epi apply " + strconv.FormatInt(id, 10) + "\n- /epi reject " + strconv.FormatInt(id, 10))
	return strings.TrimSpace(b.String())
}
