package brain

import (
	"database/sql"
	"fmt"
	"strings"
	"time"
)

type ABTrial struct {
	ID        int64
	CreatedAt string
	Prompt    string
	AModel    string
	AText     string
	BModel    string
	BText     string
	Status    string
	Choice    string
	ChosenAt  string
}

func InsertABTrial(db *sql.DB, prompt, aModel, aText, bModel, bText string) (int64, error) {
	if db == nil {
		return 0, nil
	}
	res, err := db.Exec(
		`INSERT INTO ab_trials(created_at,prompt,a_model,a_text,b_model,b_text,status,choice,chosen_at)
		 VALUES(?,?,?,?,?,?,?,'','')`,
		time.Now().Format(time.RFC3339), strings.TrimSpace(prompt), strings.TrimSpace(aModel), strings.TrimSpace(aText), strings.TrimSpace(bModel), strings.TrimSpace(bText), "open",
	)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func GetABTrial(db *sql.DB, id int64) (ABTrial, bool) {
	if db == nil || id <= 0 {
		return ABTrial{}, false
	}
	var t ABTrial
	_ = db.QueryRow(`SELECT id, created_at, prompt, a_model, a_text, b_model, b_text, status, choice, chosen_at FROM ab_trials WHERE id=?`, id).
		Scan(&t.ID, &t.CreatedAt, &t.Prompt, &t.AModel, &t.AText, &t.BModel, &t.BText, &t.Status, &t.Choice, &t.ChosenAt)
	t.Prompt = strings.TrimSpace(t.Prompt)
	t.AText = strings.TrimSpace(t.AText)
	t.BText = strings.TrimSpace(t.BText)
	t.Status = strings.TrimSpace(t.Status)
	t.Choice = strings.TrimSpace(t.Choice)
	return t, t.ID > 0
}

func ChooseABTrial(db *sql.DB, id int64, choice string) error {
	if db == nil || id <= 0 {
		return nil
	}
	choice = strings.ToLower(strings.TrimSpace(choice))
	if choice != "a" && choice != "b" && choice != "none" {
		return fmt.Errorf("choice must be a|b|none")
	}
	_, err := db.Exec(`UPDATE ab_trials SET status='chosen', choice=?, chosen_at=? WHERE id=?`, choice, time.Now().Format(time.RFC3339), id)
	return err
}

func RenderABTrial(t ABTrial) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("AB#%d\n", t.ID))
	b.WriteString("A (" + t.AModel + "):\n" + t.AText + "\n\n")
	b.WriteString("B (" + t.BModel + "):\n" + t.BText + "\n\n")
	b.WriteString("Wähle: /pick " + fmt.Sprintf("%d a|b|none", t.ID))
	return strings.TrimSpace(b.String())
}
