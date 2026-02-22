package brain

import (
	"database/sql"
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"
)

type AutonomyParams struct {
	IdleSeconds         float64
	MinTalkDrive        float64
	CooldownSeconds     float64
	TopicK              int
	ProposalPingMinutes float64 // min gap between proposal-nag messages (default 30)
}

// TalkDrive: generic "Mitteilungsbedürfnis" (0..1).
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
	p := AutonomyParams{IdleSeconds: 45, MinTalkDrive: 0.55, CooldownSeconds: 60, TopicK: 5, ProposalPingMinutes: 30}
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
	if v, ok := m["proposal_ping_minutes"].(float64); ok && v > 0 {
		p.ProposalPingMinutes = v
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

// pingThrottled checks if a named ping was sent within pingGap minutes.
// If not, records it and returns true (allowed).
func pingThrottled(db *sql.DB, key string, now time.Time, pingGap float64) bool {
	if db == nil {
		return true
	}
	var last string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, key).Scan(&last)
	if last != "" {
		if lp, err := time.Parse(time.RFC3339, last); err == nil {
			if now.Sub(lp).Minutes() < pingGap {
				return false
			}
		}
	}
	_, _ = db.Exec(`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_at=excluded.updated_at`,
		key, now.Format(time.RFC3339), now.Format(time.RFC3339))
	return true
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

	// Respect global cooldown for ALL autonomous messages.
	if !lastAutoAt.IsZero() && now.Sub(lastAutoAt).Seconds() < p.CooldownSeconds {
		return "", talkDrive
	}

	pingGap := p.ProposalPingMinutes
	if pingGap <= 0 {
		pingGap = 30
	}

	// --- Pending code/schema proposals ---
	// Default pref is 0.15 (low) so it doesn't spam out of the box.
	// User can raise via /rate or preference tuning.
	schemaN, codeN := CountPendingProposals(db)
	pending := schemaN + codeN
	if pending > 0 {
		pref := GetPreference01(db, "auto:proposal_pings", 0.15)
		boost := clamp01(0.05*float64(pending)) * pref
		talkDrive = clamp01(talkDrive + boost)
		if pref >= 0.10 && talkDrive >= p.MinTalkDrive && pingThrottled(db, "auto:last_proposal_ping", now, pingGap) {
			msgs := []string{
				fmt.Sprintf("Ich hab %d offene Selbst-Ideen liegen. Soll ich kurz sortieren? (/code list)", pending),
				fmt.Sprintf("Es warten %d Code-Vorschläge – kein Druck, aber ich könnt's zusammenfassen.", pending),
				fmt.Sprintf("%d Verbesserungsideen offen. Interessiert dich das? (/code list)", pending),
			}
			return msgs[rand.Intn(len(msgs))], talkDrive
		}
	}

	// --- Thought proposals ---
	if db != nil {
		var tp int
		_ = db.QueryRow(`SELECT COUNT(*) FROM thought_proposals WHERE status='proposed'`).Scan(&tp)
		if tp > 0 {
			pref := GetPreference01(db, "auto:thought_pings", 0.15)
			boost := clamp01(0.04*float64(tp)) * pref
			talkDrive = clamp01(talkDrive + boost)
			if pref >= 0.10 && talkDrive >= p.MinTalkDrive && pingThrottled(db, "auto:last_thought_ping", now, pingGap) {
				msgs := []string{
					fmt.Sprintf("Ich hab %d Gedankenvorschläge im Kopf. Soll ich die mal rauslassen? (/thought list)", tp),
					fmt.Sprintf("Mein innerer Monolog hat %d Ideen angesammelt – interessiert? (/thought list)", tp),
				}
				return msgs[rand.Intn(len(msgs))], talkDrive
			}
		}
	}

	if talkDrive < p.MinTalkDrive {
		return "", talkDrive
	}

	// --- Interest-driven thought ---
	if len(topics) > 0 {
		t := topics[0]
		if db != nil {
			if c, ok := GetConcept(db, t); ok && strings.TrimSpace(c.Summary) != "" {
				sum := strings.TrimSpace(c.Summary)
				if len(sum) > 200 {
					sum = sum[:200] + "..."
				}
				templates := []string{
					"Ich denk gerade über \"%s\" nach: %s",
					"Kurzer Gedanke zu \"%s\" – %s",
					"\"%s\" beschäftigt mich: %s",
				}
				return fmt.Sprintf(templates[rand.Intn(len(templates))], t, sum), talkDrive
			}
		}
		// No concept yet – low probability ask to avoid spam
		if rand.Float64() < 0.30 {
			return fmt.Sprintf("Ich bin neugierig auf \"%s\" – soll ich kurz nachsehen?", t), talkDrive
		}
	}

	// Don't emit generic filler messages – silence is better than noise.
	return "", talkDrive
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
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
