package brain

import (
	"database/sql"
	"math"
	"strings"
	"time"
	"unicode"

	"frankenstein-v0/internal/epi"
)

// NBIntent is a lightweight online Naive Bayes classifier backed by SQLite tables.
// It learns from /rate and /caught events.
type NBIntent struct {
	DB *sql.DB
}

func NewNBIntent(db *sql.DB) *NBIntent { return &NBIntent{DB: db} }

// Tokenize: simple, language-agnostic-ish (keeps letters/digits, splits on others).
func tokenize(s string) []string {
	s = strings.ToLower(s)
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		t := b.String()
		b.Reset()
		if len(t) < 2 {
			return
		}
		// tiny stop list (very small, keep generic)
		switch t {
		case "und", "oder", "der", "die", "das", "ein", "eine", "ich", "du", "wir", "ihr", "sie", "es":
			return
		}
		out = append(out, t)
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' {
			b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// ApplyFeedback updates the NB tables.
// weight can be positive (reinforce) or negative (unlearn). Counts never go below 0.
func (nb *NBIntent) ApplyFeedback(intent string, text string, weight float64) {
	if nb == nil || nb.DB == nil {
		return
	}
	intent = strings.TrimSpace(strings.ToUpper(intent))
	if intent == "" {
		return
	}
	toks := tokenize(text)
	if len(toks) == 0 {
		return
	}

	// update prior
	nb.bumpPrior(intent, weight)
	// update tokens
	for _, tok := range toks {
		nb.bumpToken(intent, tok, weight)
		nb.bumpTokenTotal(intent, weight)
	}
}

func (nb *NBIntent) bumpPrior(intent string, delta float64) {
	// read current
	var cur float64
	_ = nb.DB.QueryRow(`SELECT count FROM intent_nb_prior WHERE intent=?`, intent).Scan(&cur)
	cur += delta
	if cur < 0 {
		cur = 0
	}
	_, _ = nb.DB.Exec(
		`INSERT INTO intent_nb_prior(intent,count) VALUES(?,?)
         ON CONFLICT(intent) DO UPDATE SET count=excluded.count`,
		intent, cur,
	)
}

func (nb *NBIntent) bumpToken(intent, tok string, delta float64) {
	var cur float64
	_ = nb.DB.QueryRow(`SELECT count FROM intent_nb_token WHERE token=? AND intent=?`, tok, intent).Scan(&cur)
	cur += delta
	if cur < 0 {
		cur = 0
	}
	_, _ = nb.DB.Exec(
		`INSERT INTO intent_nb_token(token,intent,count) VALUES(?,?,?)
         ON CONFLICT(token,intent) DO UPDATE SET count=excluded.count`,
		tok, intent, cur,
	)
}

func (nb *NBIntent) bumpTokenTotal(intent string, delta float64) {
	var cur float64
	_ = nb.DB.QueryRow(`SELECT token_total FROM intent_nb_meta WHERE intent=?`, intent).Scan(&cur)
	cur += delta
	if cur < 0 {
		cur = 0
	}
	_, _ = nb.DB.Exec(
		`INSERT INTO intent_nb_meta(intent,token_total) VALUES(?,?)
         ON CONFLICT(intent) DO UPDATE SET token_total=excluded.token_total`,
		intent, cur,
	)
}

type NBPrediction struct {
	Intent string
	Prob   float64 // P(best)
}

// Predict returns best intent and confidence using multinomial NB with Laplace smoothing.
func (nb *NBIntent) Predict(text string, eg *epi.Epigenome) NBPrediction {
	if nb == nil || nb.DB == nil || eg == nil {
		return NBPrediction{}
	}
	enabled, minTok, _, alpha := eg.IntentNBParams()
	if !enabled {
		return NBPrediction{}
	}
	toks := tokenize(text)
	if len(toks) < minTok {
		return NBPrediction{}
	}

	// get intents with priors
	rows, err := nb.DB.Query(`SELECT intent, count FROM intent_nb_prior`)
	if err != nil {
		return NBPrediction{}
	}
	defer rows.Close()

	type ic struct {
		intent string
		prior  float64
	}
	var intents []ic
	totalPrior := 0.0
	for rows.Next() {
		var in string
		var c float64
		_ = rows.Scan(&in, &c)
		if strings.TrimSpace(in) == "" {
			continue
		}
		intents = append(intents, ic{intent: in, prior: c})
		totalPrior += c
	}
	if len(intents) == 0 {
		return NBPrediction{}
	}
	if totalPrior <= 0 {
		totalPrior = float64(len(intents))
		for i := range intents {
			intents[i].prior = 1
		}
	}

	// approximate vocab size (distinct tokens)
	var vocabSize float64
	_ = nb.DB.QueryRow(`SELECT COUNT(DISTINCT token) FROM intent_nb_token`).Scan(&vocabSize)
	if vocabSize < 1 {
		vocabSize = 1
	}

	// compute log scores
	logp := make([]float64, len(intents))
	maxLog := -1e18
	for i, it := range intents {
		// log prior
		lp := math.Log((it.prior + alpha) / (totalPrior + alpha*float64(len(intents))))

		// token total for intent
		var tokTotal float64
		_ = nb.DB.QueryRow(`SELECT token_total FROM intent_nb_meta WHERE intent=?`, it.intent).Scan(&tokTotal)
		den := tokTotal + alpha*vocabSize
		if den <= 0 {
			den = alpha * vocabSize
		}

		for _, tok := range toks {
			var c float64
			_ = nb.DB.QueryRow(`SELECT count FROM intent_nb_token WHERE token=? AND intent=?`, tok, it.intent).Scan(&c)
			lp += math.Log((c + alpha) / den)
		}
		logp[i] = lp
		if lp > maxLog {
			maxLog = lp
		}
	}

	// softmax to probability for best class
	sum := 0.0
	bestI := 0
	bestV := -1.0
	for i := range logp {
		v := math.Exp(logp[i] - maxLog)
		sum += v
		if v > bestV {
			bestV = v
			bestI = i
		}
	}
	if sum <= 0 {
		return NBPrediction{}
	}
	prob := bestV / sum
	return NBPrediction{Intent: intents[bestI].intent, Prob: prob}
}

// SaveReplyContext stores mapping message_id -> (user_text, intent).
func SaveReplyContext(db *sql.DB, messageID int64, userText string, intent string) {
	if db == nil || messageID <= 0 {
		return
	}
	if strings.TrimSpace(userText) == "" || strings.TrimSpace(intent) == "" {
		return
	}
	_, _ = db.Exec(
		`INSERT INTO reply_context(message_id,user_text,intent,created_at)
         VALUES(?,?,?,?)
         ON CONFLICT(message_id) DO UPDATE SET user_text=excluded.user_text, intent=excluded.intent`,
		messageID, userText, strings.ToUpper(strings.TrimSpace(intent)), time.Now().Format(time.RFC3339),
	)
}

func LoadReplyContext(db *sql.DB, messageID int64) (userText string, intent string, ok bool) {
	if db == nil || messageID <= 0 {
		return "", "", false
	}
	err := db.QueryRow(`SELECT user_text, intent FROM reply_context WHERE message_id=?`, messageID).Scan(&userText, &intent)
	if err != nil {
		return "", "", false
	}
	userText = strings.TrimSpace(userText)
	intent = strings.TrimSpace(intent)
	return userText, intent, userText != "" && intent != ""
}
