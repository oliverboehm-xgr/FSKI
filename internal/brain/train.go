package brain

import (
	"database/sql"
	"strconv"
	"strings"
	"time"
)

func InsertTrainTrial(db *sql.DB, userMsgID int64, topic, intent, ctxKey string, aAct, aSty, aTxt, bAct, bSty, bTxt string) (int64, error) {
	if db == nil {
		return 0, nil
	}
	now := time.Now().Format(time.RFC3339)
	res, err := db.Exec(`INSERT INTO train_trials(created_at,user_msg_id,topic,intent,ctx_key,a_action,a_style,a_text,b_action,b_style,b_text,chosen,note)
    VALUES(?,?,?,?,?,?,?,?,?,?,?, '', '')`,
		now, userMsgID, topic, intent, ctxKey, aAct, aSty, aTxt, bAct, bSty, bTxt)
	if err != nil {
		return 0, err
	}
	id, _ := res.LastInsertId()
	return id, nil
}

func UpdateTrainTrialNote(db *sql.DB, id int64, note string) error {
	if db == nil || id <= 0 {
		return nil
	}
	_, err := db.Exec(`UPDATE train_trials SET note=? WHERE id=?`, strings.TrimSpace(note), id)
	return err
}

type TrainTrialFull struct {
	ID        int64
	CreatedAt string
	UserMsgID int64
	Topic     string
	Intent    string
	CtxKey    string
	AAction   string
	AStyle    string
	AText     string
	BAction   string
	BStyle    string
	BText     string
	Chosen    string
	Note      string
}

func GetTrainTrialFull(db *sql.DB, id int64) (TrainTrialFull, bool) {
	if db == nil || id <= 0 {
		return TrainTrialFull{}, false
	}
	var t TrainTrialFull
	_ = db.QueryRow(`SELECT id,created_at,user_msg_id,topic,intent,ctx_key,a_action,a_style,a_text,b_action,b_style,b_text,chosen,note FROM train_trials WHERE id=?`, id).
		Scan(&t.ID, &t.CreatedAt, &t.UserMsgID, &t.Topic, &t.Intent, &t.CtxKey, &t.AAction, &t.AStyle, &t.AText, &t.BAction, &t.BStyle, &t.BText, &t.Chosen, &t.Note)
	return t, t.ID > 0
}

func ChooseTrainTrial(db *sql.DB, id int64, choice string) error {
	if db == nil || id <= 0 {
		return nil
	}
	choice = strings.ToUpper(strings.TrimSpace(choice))
	if choice != "A" && choice != "B" && choice != "NONE" {
		return nil
	}
	_, err := db.Exec(`UPDATE train_trials SET chosen=? WHERE id=?`, choice, id)
	return err
}

func GetTrainTrial(db *sql.DB, id int64) (ctxKey, aAct, aSty, bAct, bSty, chosen string, ok bool) {
	if db == nil || id <= 0 {
		return "", "", "", "", "", "", false
	}
	_ = db.QueryRow(`SELECT ctx_key,a_action,a_style,b_action,b_style,chosen FROM train_trials WHERE id=?`, id).Scan(&ctxKey, &aAct, &aSty, &bAct, &bSty, &chosen)
	ok = ctxKey != ""
	return
}

func ApplyTrainChoice(db *sql.DB, trialID int64, choice string) {
	ctxKey, aAct, aSty, bAct, bSty, _, ok := GetTrainTrial(db, trialID)
	if !ok {
		return
	}
	choice = strings.ToUpper(strings.TrimSpace(choice))
	chosenAction := ""
	if choice == "A" {
		chosenAction = aAct
		// If A/B are identical on an axis, do not update that axis (prevents double-counting noise).
		if aAct != "" && bAct != "" && aAct != bAct {
			UpdatePolicy(db, ctxKey, aAct, 1.0)
			UpdatePolicy(db, ctxKey, bAct, 0.0)
			UpdatePreferenceEMA(db, "strat:"+aAct, 1.0, 0.12)
			UpdatePreferenceEMA(db, "strat:"+bAct, -0.7, 0.12)
		}
		if aSty != "" && bSty != "" && aSty != bSty {
			UpdatePreferenceEMA(db, "style:"+aSty, 1.0, 0.12)
			UpdatePreferenceEMA(db, "style:"+bSty, -0.7, 0.12)
		}
	} else if choice == "B" {
		chosenAction = bAct
		if aAct != "" && bAct != "" && aAct != bAct {
			UpdatePolicy(db, ctxKey, bAct, 1.0)
			UpdatePolicy(db, ctxKey, aAct, 0.0)
			UpdatePreferenceEMA(db, "strat:"+bAct, 1.0, 0.12)
			UpdatePreferenceEMA(db, "strat:"+aAct, -0.7, 0.12)
		}
		if aSty != "" && bSty != "" && aSty != bSty {
			UpdatePreferenceEMA(db, "style:"+bSty, 1.0, 0.12)
			UpdatePreferenceEMA(db, "style:"+aSty, -0.7, 0.12)
		}
	}

	// collect LoRA preference sample (chosen vs rejected)
	InsertLoRASampleFromTrainTrial(db, trialID, choice)

	applySoftWeightMutation(db, ctxKey, chosenAction)
}

func applySoftWeightMutation(db *sql.DB, ctxKey, chosenAction string) {
	if db == nil || strings.TrimSpace(ctxKey) == "" || strings.TrimSpace(chosenAction) == "" {
		return
	}
	rate := kvFloat(db, "train:soft_weight_mutation", 0.03)
	if rate < 0.0 {
		rate = 0.0
	}
	if rate > 0.15 {
		rate = 0.15
	}
	if rate == 0 {
		return
	}
	for _, act := range DefaultPolicyActions {
		if strings.TrimSpace(act) == "" {
			continue
		}
		reward := 0.5
		if act == chosenAction {
			reward = 0.5 + rate
		} else {
			reward = 0.5 - (rate / float64(maxInt(1, len(DefaultPolicyActions)-1)))
		}
		UpdatePolicy(db, ctxKey, act, reward)
	}
}

func kvFloat(db *sql.DB, key string, fallback float64) float64 {
	if db == nil {
		return fallback
	}
	var raw string
	if err := db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, strings.TrimSpace(key)).Scan(&raw); err != nil {
		return fallback
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return fallback
	}
	return v
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
