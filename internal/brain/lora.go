package brain

import (
	"bufio"
	"database/sql"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// LoRASample is a pairwise preference sample (chosen vs rejected) for LoRA/DPO training.
type LoRASample struct {
	ID        int64
	CreatedAt string
	Prompt    string
	Chosen    string
	Rejected  string
	MetaJSON  string
}

// LoRAJob represents an external training run request.
type LoRAJob struct {
	ID          int64
	CreatedAt   string
	Status      string // queued|running|done|error
	BaseModel   string
	DatasetPath string
	OutDir      string
	Notes       string
	UpdatedAt   string
}

func InsertLoRASample(db *sql.DB, prompt, chosen, rejected, metaJSON string) {
	if db == nil {
		return
	}
	prompt = strings.TrimSpace(prompt)
	chosen = strings.TrimSpace(chosen)
	rejected = strings.TrimSpace(rejected)
	metaJSON = strings.TrimSpace(metaJSON)
	if chosen == "" || rejected == "" {
		return
	}
	_, _ = db.Exec(`INSERT INTO lora_samples(created_at,prompt,chosen,rejected,meta_json) VALUES(?,?,?,?,?)`,
		time.Now().Format(time.RFC3339), prompt, chosen, rejected, metaJSON)
}

// InsertLoRASampleFromTrainTrial stores a preference sample from a train_trials choice (A vs B).
func InsertLoRASampleFromTrainTrial(db *sql.DB, trialID int64, choice string) {
	if db == nil || trialID <= 0 {
		return
	}
	t, ok := GetTrainTrialFull(db, trialID)
	if !ok {
		return
	}
	choice = strings.ToUpper(strings.TrimSpace(choice))
	var chosen, rej string
	if choice == "A" {
		chosen = t.AText
		rej = t.BText
	} else if choice == "B" {
		chosen = t.BText
		rej = t.AText
	} else {
		return
	}
	meta := map[string]any{
		"trial_id": trialID,
		"ctx":      strings.TrimSpace(t.CtxKey),
		"topic":    strings.TrimSpace(t.Topic),
		"intent":   strings.TrimSpace(t.Intent),
		"a_action": strings.TrimSpace(t.AAction),
		"b_action": strings.TrimSpace(t.BAction),
		"a_style":  strings.TrimSpace(t.AStyle),
		"b_style":  strings.TrimSpace(t.BStyle),
	}
	b, _ := json.Marshal(meta)
	InsertLoRASample(db, "TRAIN_TRIAL topic="+strings.TrimSpace(t.Topic)+" intent="+strings.TrimSpace(t.Intent), chosen, rej, string(b))
}

func ListLoRASamples(db *sql.DB, limit int) ([]LoRASample, error) {
	if db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 20
	}
	if limit > 200 {
		limit = 200
	}
	rows, err := db.Query(`SELECT id,created_at,prompt,chosen,rejected,meta_json FROM lora_samples ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []LoRASample
	for rows.Next() {
		var s LoRASample
		_ = rows.Scan(&s.ID, &s.CreatedAt, &s.Prompt, &s.Chosen, &s.Rejected, &s.MetaJSON)
		s.Prompt = strings.TrimSpace(s.Prompt)
		s.Chosen = strings.TrimSpace(s.Chosen)
		s.Rejected = strings.TrimSpace(s.Rejected)
		s.MetaJSON = strings.TrimSpace(s.MetaJSON)
		out = append(out, s)
	}
	return out, nil
}

func ExportLoRASamplesJSONL(db *sql.DB, limit int, outPath string) (int, error) {
	if db == nil {
		return 0, errors.New("db nil")
	}
	outPath = strings.TrimSpace(outPath)
	if outPath == "" {
		return 0, errors.New("missing path")
	}
	samples, err := ListLoRASamples(db, limit)
	if err != nil {
		return 0, err
	}
	if len(samples) == 0 {
		return 0, errors.New("no samples")
	}
	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return 0, err
	}
	f, err := os.Create(outPath)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	defer w.Flush()

	type rec struct {
		Prompt   string `json:"prompt"`
		Chosen   string `json:"chosen"`
		Rejected string `json:"rejected"`
		Meta     string `json:"meta,omitempty"`
	}
	for _, s := range samples {
		r := rec{Prompt: s.Prompt, Chosen: s.Chosen, Rejected: s.Rejected, Meta: s.MetaJSON}
		b, _ := json.Marshal(r)
		_, _ = w.Write(b)
		_, _ = w.WriteString("\n")
	}
	return len(samples), nil
}

func QueueLoRAJob(db *sql.DB, baseModel string, datasetPath string, outDir string, notes string) (int64, error) {
	if db == nil {
		return 0, errors.New("db nil")
	}
	baseModel = strings.TrimSpace(baseModel)
	datasetPath = strings.TrimSpace(datasetPath)
	outDir = strings.TrimSpace(outDir)
	if baseModel == "" || datasetPath == "" || outDir == "" {
		return 0, errors.New("missing args")
	}
	_, err := os.Stat(datasetPath)
	if err != nil {
		return 0, err
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return 0, err
	}
	now := time.Now().Format(time.RFC3339)
	res, err := db.Exec(`INSERT INTO lora_jobs(created_at,status,base_model,dataset_path,out_dir,notes,updated_at) VALUES(?,?,?,?,?,?,?)`,
		now, "queued", baseModel, datasetPath, outDir, strings.TrimSpace(notes), now)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func ListLoRAJobs(db *sql.DB, limit int) ([]LoRAJob, error) {
	if db == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = 25
	}
	rows, err := db.Query(`SELECT id,created_at,status,base_model,dataset_path,out_dir,notes,updated_at FROM lora_jobs ORDER BY id DESC LIMIT ?`, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []LoRAJob
	for rows.Next() {
		var j LoRAJob
		_ = rows.Scan(&j.ID, &j.CreatedAt, &j.Status, &j.BaseModel, &j.DatasetPath, &j.OutDir, &j.Notes, &j.UpdatedAt)
		j.Status = strings.TrimSpace(j.Status)
		j.BaseModel = strings.TrimSpace(j.BaseModel)
		j.DatasetPath = strings.TrimSpace(j.DatasetPath)
		j.OutDir = strings.TrimSpace(j.OutDir)
		j.Notes = strings.TrimSpace(j.Notes)
		out = append(out, j)
	}
	return out, nil
}

func RunLoRAJob(db *sql.DB, jobID int64) (string, error) {
	if db == nil || jobID <= 0 {
		return "", errors.New("bad job id")
	}
	var j LoRAJob
	err := db.QueryRow(`SELECT id,created_at,status,base_model,dataset_path,out_dir,notes,updated_at FROM lora_jobs WHERE id=?`, jobID).
		Scan(&j.ID, &j.CreatedAt, &j.Status, &j.BaseModel, &j.DatasetPath, &j.OutDir, &j.Notes, &j.UpdatedAt)
	if err != nil {
		return "", err
	}
	if strings.TrimSpace(j.Status) == "running" {
		return "already running", nil
	}
	cmdT := kvString(db, "lora:trainer_cmd", "")
	if strings.TrimSpace(cmdT) == "" {
		return "", errors.New("kv_state missing lora:trainer_cmd")
	}
	// expand placeholders
	dataset := j.DatasetPath
	out := j.OutDir
	cmdLine := strings.ReplaceAll(cmdT, "{base}", j.BaseModel)
	cmdLine = strings.ReplaceAll(cmdLine, "{dataset}", dataset)
	cmdLine = strings.ReplaceAll(cmdLine, "{out}", out)

	now := time.Now().Format(time.RFC3339)
	_, _ = db.Exec(`UPDATE lora_jobs SET status=?, updated_at=? WHERE id=?`, "running", now, jobID)

	c := exec.Command("bash", "-lc", cmdLine)
	c.Env = os.Environ()
	b, runErr := c.CombinedOutput()
	log := strings.TrimSpace(string(b))
	status := "done"
	if runErr != nil {
		status = "error"
		log = log + "\nERR: " + runErr.Error()
	}
	_, _ = db.Exec(`UPDATE lora_jobs SET status=?, updated_at=?, notes=? WHERE id=?`, status, time.Now().Format(time.RFC3339), clipForContext(j.Notes+"\n"+log, 4000), jobID)
	return log, runErr
}

func kvString(db *sql.DB, key string, fallback string) string {
	if db == nil {
		return fallback
	}
	var raw string
	if err := db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, strings.TrimSpace(key)).Scan(&raw); err != nil {
		return fallback
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return fallback
	}
	return raw
}
