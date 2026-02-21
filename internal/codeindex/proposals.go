package codeindex

import (
	"database/sql"
	"time"
)

func SaveProposal(db *sql.DB, title string, diff string, notes string) (int64, error) {
	if db == nil {
		return 0, nil
	}
	res, err := db.Exec(
		`INSERT INTO code_proposals(created_at,title,diff,status,notes) VALUES(?,?,?,?,?)`,
		time.Now().Format(time.RFC3339), title, diff, "proposed", notes,
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}
