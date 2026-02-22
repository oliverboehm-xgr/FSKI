package brain

import (
	"database/sql"
	"math"
	"strings"
	"time"
)

type AutonomyParams struct {
	IdleSeconds     float64
	MinTalkDrive    float64
	CooldownSeconds float64
	TopicK          int
}

// TalkDrive: generic “Mitteilungsbedürfnis” (0..1).
// No hardcoded greetings. Uses time-since-user + curiosity + negative affect inhibition.
func ComputeTalkDrive(curiosity float64, idleSec float64, aff *AffectState) float64 {
	tau := 120.0
	socialNeed := 1.0 - math.Exp(-idleSec/tau)

	inhib := 0.0
	if aff != nil {
		inhib += 0.9 * aff.Get("shame")
		inhib += 0.6 * aff.Get("fear")
		inhib += 0.4 * aff.Get("pain")
		inhib += 0.3 * aff.Get("unwell")
		inhib += 0.2 * aff.Get("sorrow")
	}

	td := 0.15 + 0.70*socialNeed + 0.25*clamp01(curiosity) - 0.80*clamp01(inhib)
	return clamp01(td)
}

func LoadAutonomyParams(eg interface {
	ModuleEnabled(string) bool
	ModuleParams(string) map[string]any
}) AutonomyParams {
	p := AutonomyParams{IdleSeconds: 45, MinTalkDrive: 0.55, CooldownSeconds: 60, TopicK: 5}
	if eg == nil || !eg.ModuleEnabled("autonomy") {
		return p
	}
	m := eg.ModuleParams("autonomy")
	if v, ok := m["idle_seconds"].(float64); ok && v > 5 {
		p.IdleSeconds = v
	}
	if v, ok := m["min_talk_drive"].(float64); ok {
		p.MinTalkDrive = clamp01(v)
	}
	if v, ok := m["cooldown_seconds"].(float64); ok && v > 5 {
		p.CooldownSeconds = v
	}
	if v, ok := m["topic_k"].(float64); ok && int(v) > 0 {
		p.TopicK = int(v)
	}
	return p
}

func TopInterests(db *sql.DB, k int) ([]string, error) {
	if db == nil {
		return nil, nil
	}
	if k <= 0 {
		k = 5
	}
	rows, err := db.Query(`SELECT topic FROM interests ORDER BY weight DESC LIMIT ?`, k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []string
	for rows.Next() {
		var t string
		_ = rows.Scan(&t)
		if t != "" {
			out = append(out, t)
		}
	}
	return out, nil
}

func LastUserMessageAt(db *sql.DB) time.Time {
	if db == nil {
		return time.Time{}
	}
	var ts string
	_ = db.QueryRow(`SELECT m.created_at FROM messages m JOIN message_meta mm ON mm.message_id=m.id WHERE mm.kind='user' ORDER BY m.id DESC LIMIT 1`).Scan(&ts)
	t, _ := time.Parse(time.RFC3339, ts)
	return t
}

// TickAutonomy returns a spontaneous message or "".
func TickAutonomy(db *sql.DB, now time.Time, lastUserAt time.Time, lastAutoAt time.Time, curiosity float64, aff *AffectState, topics []string, p AutonomyParams) (msg string, talkDrive float64) {
	idle := now.Sub(lastUserAt).Seconds()
	if idle < 0 {
		idle = 0
	}
	if idle < p.IdleSeconds {
		return "", ComputeTalkDrive(curiosity, idle, aff)
	}
	talkDrive = ComputeTalkDrive(curiosity, idle, aff)

	// Boost talk drive if there are pending proposals.
	schemaN, codeN := CountPendingProposals(db)
	pending := schemaN + codeN
	if pending > 0 {
		boost := 0.08 * float64(pending)
		if boost > 0.35 {
			boost = 0.35
		}
		talkDrive = clamp01(talkDrive + boost)
		if talkDrive >= p.MinTalkDrive {
			return "Ich habe offene Vorschläge zur Selbstverbesserung (Schema: " + itoa(schemaN) + ", Code: " + itoa(codeN) + "). Soll ich sie kurz zusammenfassen? (Commands: /schema list | /schema show <id> | /schema apply <id> | /code list | /code show <id>)", talkDrive
		}
	}

	if db != nil {
		var tp int
		_ = db.QueryRow(`SELECT COUNT(*) FROM thought_proposals WHERE status='proposed'`).Scan(&tp)
		if tp > 0 {
			boost := 0.06 * float64(tp)
			if boost > 0.25 {
				boost = 0.25
			}
			talkDrive = clamp01(talkDrive + boost)
			if talkDrive >= p.MinTalkDrive {
				return "Ich habe " + itoa(tp) + " offene Selbstverbesserungs-Ideen aus meiner Gedankenwelt (thought_proposals). Soll ich sie anzeigen oder materialisieren? (Commands: /thought list | /thought show <id> | /thought materialize <id|all>)", talkDrive
			}
		}
	}

	if !lastAutoAt.IsZero() && now.Sub(lastAutoAt).Seconds() < p.CooldownSeconds {
		return "", talkDrive
	}
	if talkDrive < p.MinTalkDrive {
		return "", talkDrive
	}

	if len(topics) > 0 {
		t := topics[0]
		if db != nil {
			if c, ok := GetConcept(db, t); ok && strings.TrimSpace(c.Summary) != "" {
				sum := strings.TrimSpace(c.Summary)
				if len(sum) > 220 {
					sum = sum[:220] + "..."
				}
				return "Kurzer Gedanke zu \"" + t + "\": " + sum + "\nSoll ich dazu weiter scouten (Web) oder willst du die Richtung vorgeben?", talkDrive
			}
		}
		return "Ich hab gerade Mitteilungsdrang zu \"" + t + "\". Soll ich kurz scouten (Web) oder willst du ein anderes Thema setzen?", talkDrive
	}

	return "Ich hab gerade Lust auf ein kurzes Gespräch. Willst du ein Thema setzen – oder soll ich eins vorschlagen?", talkDrive
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var b [32]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + (n % 10))
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
