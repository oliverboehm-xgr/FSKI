package brain

import (
	"database/sql"
	"encoding/json"
	"time"
)

type Candidate struct {
	ID       string
	Yields   []string
	Prereq   []string
	Cost     float64
	Evidence float64
	Helps    map[string]float64
}

func EnsureDefaultCandidates(db *sql.DB) {
	if db == nil {
		return
	}
	now := time.Now().Format(time.RFC3339)
	def := []Candidate{
		{ID: "expand:disk:add_path", Yields: []string{"disk:NEW_PATH"}, Prereq: []string{"user_action:add_storage_path"}, Cost: 0.35, Evidence: 0.35, Helps: map[string]float64{"survival": 0.7}},
		{ID: "expand:disk:cleanup", Yields: []string{"disk:C:\\"}, Prereq: []string{"user_action:cleanup_disk"}, Cost: 0.20, Evidence: 0.55, Helps: map[string]float64{"survival": 0.8}},
		{ID: "expand:ram:free", Yields: []string{"ram"}, Prereq: []string{"user_action:close_apps"}, Cost: 0.15, Evidence: 0.60, Helps: map[string]float64{"survival": 0.7}},
		{ID: "expand:ram:upgrade", Yields: []string{"ram"}, Prereq: []string{"hardware_purchase:ram"}, Cost: 0.70, Evidence: 0.50, Helps: map[string]float64{"survival": 0.9}},
		{ID: "expand:sensor:camera", Yields: []string{"sensor:camera"}, Prereq: []string{"user_action:provide_camera", "permission:camera", "adapter_needed"}, Cost: 0.55, Evidence: 0.25, Helps: map[string]float64{"social": 0.7, "curiosity": 0.3}},
	}
	for _, c := range def {
		y, _ := json.Marshal(c.Yields)
		p, _ := json.Marshal(c.Prereq)
		h, _ := json.Marshal(c.Helps)
		_, _ = db.Exec(`INSERT INTO expand_candidates(id,yields_json,prereq_json,cost,evidence,helps_json,updated_at) VALUES(?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET updated_at=excluded.updated_at`, c.ID, string(y), string(p), c.Cost, c.Evidence, string(h), now)
	}
}

func LoadCandidates(db *sql.DB) ([]Candidate, error) {
	rows, err := db.Query(`SELECT id,yields_json,prereq_json,cost,evidence,helps_json FROM expand_candidates`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Candidate
	for rows.Next() {
		var c Candidate
		var y, p, h string
		_ = rows.Scan(&c.ID, &y, &p, &c.Cost, &c.Evidence, &h)
		_ = json.Unmarshal([]byte(y), &c.Yields)
		_ = json.Unmarshal([]byte(p), &c.Prereq)
		_ = json.Unmarshal([]byte(h), &c.Helps)
		out = append(out, c)
	}
	return out, nil
}

func LogCandidate(db *sql.DB, id, outcome, note string) {
	if db == nil {
		return
	}
	_, _ = db.Exec(`INSERT INTO candidate_history(created_at,candidate_id,outcome,note) VALUES(?,?,?,?)`, time.Now().Format(time.RFC3339), id, outcome, note)
}
