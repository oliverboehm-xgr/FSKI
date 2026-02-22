package brain

import (
	"database/sql"
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

func ChooseTrainTrial(db *sql.DB, id int64, choice string) error {
	if db == nil || id <= 0 {
		return nil
	}
	choice = strings.ToUpper(strings.TrimSpace(choice))
	if choice != "A" && choice != "B" {
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
	if choice == "A" {
		UpdatePolicy(db, ctxKey, aAct, 1.0)
		UpdatePolicy(db, ctxKey, bAct, 0.0)
		UpdatePreferenceEMA(db, "style:"+aSty, 1.0, 0.12)
		UpdatePreferenceEMA(db, "style:"+bSty, -0.7, 0.12)
		UpdatePreferenceEMA(db, "strat:"+aAct, 1.0, 0.12)
		UpdatePreferenceEMA(db, "strat:"+bAct, -0.7, 0.12)
	} else if choice == "B" {
		UpdatePolicy(db, ctxKey, bAct, 1.0)
		UpdatePolicy(db, ctxKey, aAct, 0.0)
		UpdatePreferenceEMA(db, "style:"+bSty, 1.0, 0.12)
		UpdatePreferenceEMA(db, "style:"+aSty, -0.7, 0.12)
		UpdatePreferenceEMA(db, "strat:"+bAct, 1.0, 0.12)
		UpdatePreferenceEMA(db, "strat:"+aAct, -0.7, 0.12)
	}
}
