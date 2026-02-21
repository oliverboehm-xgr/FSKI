package brain

import (
	"database/sql"
	"strconv"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
)

type ScoutRequest struct {
	Topic string
	Query string
}

func getKV(db *sql.DB, key string) string {
	var v string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, key).Scan(&v)
	return v
}

func setKV(db *sql.DB, key, value string) {
	_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?)
		ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		key, value, time.Now().Format(time.RFC3339))
}

func MaybeQueueScout(db *sql.DB, eg *epi.Epigenome, ws *Workspace, dr *Drives) (bool, ScoutRequest) {
	if db == nil || eg == nil || ws == nil || dr == nil {
		return false, ScoutRequest{}
	}
	intervalSec, minCur, maxPerHour, enabled := eg.ScoutParams()
	if !enabled || dr.Curiosity < minCur {
		return false, ScoutRequest{}
	}
	topic := strings.TrimSpace(ws.ActiveTopic)
	if topic == "" {
		topic = strings.TrimSpace(ws.LastTopic)
	}
	if topic == "" {
		return false, ScoutRequest{}
	}

	key := "scout_last_" + topic
	if lastStr := getKV(db, key); lastStr != "" {
		if ts, err := time.Parse(time.RFC3339, lastStr); err == nil && time.Since(ts) < time.Duration(intervalSec)*time.Second {
			return false, ScoutRequest{}
		}
	}

	hourKey := "scout_count_hour_" + time.Now().Format("2006010215")
	cnt, _ := strconv.Atoi(getKV(db, hourKey))
	if cnt >= maxPerHour {
		return false, ScoutRequest{}
	}

	if c, ok := GetConcept(db, topic); !ok || c.Confidence < 0.55 {
		setKV(db, key, time.Now().Format(time.RFC3339))
		setKV(db, hourKey, strconv.Itoa(cnt+1))
		return true, ScoutRequest{Topic: topic, Query: topic}
	}
	return false, ScoutRequest{}
}
