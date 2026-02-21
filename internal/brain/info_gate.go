package brain

import (
	"database/sql"
	"math"
	"strconv"
	"strings"
	"time"
	"unicode"

	"frankenstein-v0/internal/epi"
)

type InfoResult struct {
	Score         float64
	Tokens        []string
	ContentTokens []string
	MaxIDF        float64
	TopToken      string
	Docs          int64
}

func tokenizeGeneric(s string) []string {
	s = strings.ToLower(strings.TrimSpace(s))
	var out []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		t := b.String()
		b.Reset()
		if len(t) == 0 {
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

func getKVInt64(db *sql.DB, key string) int64 {
	var v string
	_ = db.QueryRow(`SELECT value FROM kv_state WHERE key=?`, key).Scan(&v)
	if v == "" {
		return 0
	}
	n, _ := strconv.ParseInt(v, 10, 64)
	return n
}

func setKVInt64(db *sql.DB, key string, n int64) {
	if db == nil {
		return
	}
	_, _ = db.Exec(
		`INSERT INTO kv_state(key,value,updated_at) VALUES(?,?,?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		key, strconv.FormatInt(n, 10), time.Now().Format(time.RFC3339),
	)
}

// ObserveUtterance updates token_df and doc count (N) once per user utterance.
func ObserveUtterance(db *sql.DB, text string) {
	if db == nil {
		return
	}
	toks := tokenizeGeneric(text)
	if len(toks) == 0 {
		return
	}
	// unique tokens per utterance for DF
	seen := map[string]struct{}{}
	for _, t := range toks {
		t = strings.TrimSpace(t)
		if t == "" {
			continue
		}
		seen[t] = struct{}{}
	}
	if len(seen) == 0 {
		return
	}
	N := getKVInt64(db, "token_df_docs")
	N++
	setKVInt64(db, "token_df_docs", N)

	for tok := range seen {
		var df int64
		_ = db.QueryRow(`SELECT df FROM token_df WHERE token=?`, tok).Scan(&df)
		df++
		_, _ = db.Exec(`INSERT INTO token_df(token,df) VALUES(?,?)
            ON CONFLICT(token) DO UPDATE SET df=excluded.df`, tok, df)
	}
}

// ScoreUtterance computes informativeness using learned DF/IDF.
// Does NOT update token_df (call ObserveUtterance separately).
func ScoreUtterance(db *sql.DB, eg *epi.Epigenome, text string) InfoResult {
	var res InfoResult
	text = strings.TrimSpace(text)
	if db == nil || eg == nil || text == "" {
		return res
	}
	enabled, _, idfTh, idf2Th, stopRatio, minTok, warmupMinDocs, stopMinDf := eg.InfoGateParams()
	if !enabled {
		res.Score = 1.0
		return res
	}
	if strings.HasPrefix(text, "/") {
		// commands are not "conversation" – treat as informative enough
		res.Score = 1.0
		return res
	}

	toks := tokenizeGeneric(text)
	res.Tokens = toks
	if len(toks) < minTok {
		return res
	}

	N := getKVInt64(db, "token_df_docs")
	if N < 1 {
		N = 1
	}
	res.Docs = N

	// length factor saturates at ~9 tokens, but 1 token still gets some weight
	lengthFactor := math.Log(float64(len(toks))+1.0) / math.Log(10.0)
	lengthFactor = clamp01(lengthFactor)

	// compute IDF per token
	maxIDF := 0.0
	topTok := ""
	content := make([]string, 0, len(toks))

	for _, tok := range toks {
		if tok == "" {
			continue
		}
		var df int64
		_ = db.QueryRow(`SELECT df FROM token_df WHERE token=?`, tok).Scan(&df)
		if df < 0 {
			df = 0
		}
		dfRatio := float64(df) / float64(N)
		// learned stopword suppression (after warmup + only truly frequent tokens)
		if int(N) >= warmupMinDocs && int(df) >= stopMinDf && dfRatio >= stopRatio {
			continue
		}
		idf := math.Log(float64(N+1) / float64(df+1))
		if idf > maxIDF {
			maxIDF = idf
			topTok = tok
		}

		// content token rule:
		// - length >=3 and idf>0
		// - OR rare 2-char token with very high idf (AI/VW style), generic.
		if len(tok) >= 3 {
			content = append(content, tok)
		} else if len(tok) == 2 && idf >= idf2Th {
			content = append(content, tok)
		}
	}

	res.ContentTokens = content
	res.MaxIDF = maxIDF
	res.TopToken = topTok

	if len(toks) == 0 {
		return res
	}
	contentRatio := float64(len(content)) / float64(len(toks))
	base := contentRatio * lengthFactor

	// booster for strong maxIDF (lets 1-word "Ukraine" pass, blocks greetings which lose DF quickly)
	boost := 0.0
	if maxIDF > idfTh {
		boost = 0.20 * clamp01((maxIDF-idfTh)/3.0)
	}
	res.Score = clamp01(base + boost)
	return res
}

func IsLowInfo(db *sql.DB, eg *epi.Epigenome, text string) (low bool, info InfoResult) {
	enabled, minInfo, _, _, _, _, _, _ := eg.InfoGateParams()
	if !enabled {
		return false, InfoResult{Score: 1.0}
	}
	info = ScoreUtterance(db, eg, text)
	return info.Score < minInfo, info
}
