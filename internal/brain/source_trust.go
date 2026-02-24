package brain

import (
	"database/sql"
	"net/url"
	"sort"
	"strings"
	"time"

	"frankenstein-v0/internal/websense"
)

func ensureSourceTrustTable(db *sql.DB) {
	if db == nil {
		return
	}
	_, _ = db.Exec(`
CREATE TABLE IF NOT EXISTS source_trust(
  domain TEXT PRIMARY KEY,
  score REAL NOT NULL DEFAULT 0,
  good_count INTEGER NOT NULL DEFAULT 0,
  bad_count INTEGER NOT NULL DEFAULT 0,
  updated_at TEXT NOT NULL
);`)
}

func domainFromURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return ""
	}
	pu, err := url.Parse(raw)
	if err != nil {
		return ""
	}
	return strings.ToLower(strings.TrimSpace(pu.Hostname()))
}

func GetSourceTrust(db *sql.DB, domain string) float64 {
	if db == nil {
		return 0
	}
	ensureSourceTrustTable(db)
	domain = strings.ToLower(strings.TrimSpace(domain))
	if domain == "" {
		return 0
	}
	var v float64
	_ = db.QueryRow(`SELECT score FROM source_trust WHERE domain=?`, domain).Scan(&v)
	return v
}

func UpdateSourceTrust(db *sql.DB, domain string, success bool) {
	if db == nil {
		return
	}
	ensureSourceTrustTable(db)
	domain = strings.ToLower(strings.TrimSpace(domain))
	if domain == "" {
		return
	}
	now := time.Now().Format(time.RFC3339)
	delta := 0.10
	good := 1
	bad := 0
	if !success {
		delta = -0.05
		good = 0
		bad = 1
	}
	_, _ = db.Exec(`
INSERT INTO source_trust(domain,score,good_count,bad_count,updated_at)
VALUES(?,?,?,?,?)
ON CONFLICT(domain) DO UPDATE SET
  score=source_trust.score+excluded.score,
  good_count=source_trust.good_count+excluded.good_count,
  bad_count=source_trust.bad_count+excluded.bad_count,
  updated_at=excluded.updated_at
`, domain, delta, good, bad, now)
}

// PickEvidenceResults ranks by domain trust and enforces domain diversity first.
// This is the minimal "quality" lever that later spidering/cross-checking can build upon.
func PickEvidenceResults(db *sql.DB, results []websense.SearchResult, topN int) []websense.SearchResult {
	if topN <= 0 {
		topN = 2
	}
	type scored struct {
		r      websense.SearchResult
		domain string
		score  float64
	}
	sc := make([]scored, 0, len(results))
	for _, r := range results {
		d := domainFromURL(r.URL)
		s := GetSourceTrust(db, d)
		sc = append(sc, scored{r: r, domain: d, score: s})
	}
	sort.Slice(sc, func(i, j int) bool {
		if sc[i].score == sc[j].score {
			// tie-breaker: prefer longer snippet/title (often more descriptive)
			li := len(strings.TrimSpace(sc[i].r.Snippet)) + len(strings.TrimSpace(sc[i].r.Title))
			lj := len(strings.TrimSpace(sc[j].r.Snippet)) + len(strings.TrimSpace(sc[j].r.Title))
			return li > lj
		}
		return sc[i].score > sc[j].score
	})
	out := make([]websense.SearchResult, 0, topN)
	seen := map[string]bool{}
	for _, s := range sc {
		if len(out) >= topN {
			break
		}
		if s.domain != "" && seen[s.domain] {
			continue
		}
		out = append(out, s.r)
		if s.domain != "" {
			seen[s.domain] = true
		}
	}
	// if not enough diverse domains, fill up
	if len(out) < topN {
		for _, s := range sc {
			if len(out) >= topN {
				break
			}
			out = append(out, s.r)
		}
	}
	return out
}
