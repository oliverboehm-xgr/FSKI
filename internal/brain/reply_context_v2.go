package brain

import (
	"database/sql"
	"strings"
	"time"
)

func SaveReplyContextV2(db *sql.DB, messageID int64, userText, intentMode, policyCtx, action, style string) {
	if db == nil || messageID <= 0 {
		return
	}
	userText = strings.TrimSpace(userText)
	intentMode = strings.TrimSpace(strings.ToUpper(intentMode))
	policyCtx = strings.TrimSpace(policyCtx)
	action = strings.TrimSpace(action)
	style = strings.TrimSpace(style)
	if userText == "" || intentMode == "" || policyCtx == "" || action == "" {
		return
	}
	_, _ = db.Exec(
		`INSERT INTO reply_context_v2(message_id,user_text,intent,policy_ctx,action,style,created_at)
		 VALUES(?,?,?,?,?,?,?)
		 ON CONFLICT(message_id) DO UPDATE SET user_text=excluded.user_text, intent=excluded.intent, policy_ctx=excluded.policy_ctx, action=excluded.action, style=excluded.style`,
		messageID, userText, intentMode, policyCtx, action, style, time.Now().Format(time.RFC3339),
	)
}

func LoadReplyContextV2(db *sql.DB, messageID int64) (userText, intentMode, policyCtx, action, style string, ok bool) {
	if db == nil || messageID <= 0 {
		return "", "", "", "", "", false
	}
	err := db.QueryRow(`SELECT user_text,intent,policy_ctx,action,style FROM reply_context_v2 WHERE message_id=?`, messageID).
		Scan(&userText, &intentMode, &policyCtx, &action, &style)
	if err != nil {
		return "", "", "", "", "", false
	}
	userText = strings.TrimSpace(userText)
	intentMode = strings.TrimSpace(intentMode)
	policyCtx = strings.TrimSpace(policyCtx)
	action = strings.TrimSpace(action)
	style = strings.TrimSpace(style)
	return userText, intentMode, policyCtx, action, style, userText != "" && action != ""
}
