package brain

import (
	"database/sql"
	"time"

	"frankenstein-v0/internal/epi"
)

type Traits struct {
	BluffRate     float64
	HonestyBias   float64
	SearchK       int
	FetchAttempts int
	TalkBias      float64 // 0..1, how likely to "want to share" (drives coupling)
	ResearchBias  float64 // 0..1, how eager to use senses when uncertain
}

func LoadOrInitTraits(db *sql.DB) (*Traits, error) {
	tr := &Traits{
		BluffRate:     0.08,
		HonestyBias:   0.80,
		SearchK:       8,
		FetchAttempts: 4,
		TalkBias:      0.45,
		ResearchBias:  0.55,
	}

	rows, err := db.Query(`SELECT key, value FROM traits`)
	if err != nil {
		// keep defaults
		_ = saveTrait(db, "bluff_rate", tr.BluffRate)
		_ = saveTrait(db, "honesty_bias", tr.HonestyBias)
		_ = saveTrait(db, "search_k", float64(tr.SearchK))
		_ = saveTrait(db, "fetch_attempts", float64(tr.FetchAttempts))
		_ = saveTrait(db, "talk_bias", tr.TalkBias)
		_ = saveTrait(db, "research_bias", tr.ResearchBias)
		return tr, nil
	}
	defer rows.Close()

	for rows.Next() {
		var k string
		var v float64
		_ = rows.Scan(&k, &v)
		switch k {
		case "bluff_rate":
			tr.BluffRate = clamp01(v)
		case "honesty_bias":
			tr.HonestyBias = clamp01(v)
		case "search_k":
			if v >= 1 {
				tr.SearchK = int(v)
			}
		case "fetch_attempts":
			if v >= 1 {
				tr.FetchAttempts = int(v)
			}
		case "talk_bias":
			tr.TalkBias = clamp01(v)
		case "research_bias":
			tr.ResearchBias = clamp01(v)
		}
	}

	// clamp sensible bounds
	if tr.SearchK < 4 {
		tr.SearchK = 4
	}
	if tr.SearchK > 12 {
		tr.SearchK = 12
	}
	if tr.FetchAttempts < 2 {
		tr.FetchAttempts = 2
	}
	if tr.FetchAttempts > 8 {
		tr.FetchAttempts = 8
	}

	_ = saveTrait(db, "bluff_rate", tr.BluffRate)
	_ = saveTrait(db, "honesty_bias", tr.HonestyBias)
	_ = saveTrait(db, "search_k", float64(tr.SearchK))
	_ = saveTrait(db, "fetch_attempts", float64(tr.FetchAttempts))
	_ = saveTrait(db, "talk_bias", tr.TalkBias)
	_ = saveTrait(db, "research_bias", tr.ResearchBias)
	return tr, nil
}

// ApplyRating: learning via your reactions (no hard output rules).
// Downvote => invest more in sensing next time (search deeper + more fetch attempts).
// Upvote   => become more efficient again.
func ApplyRating(db *sql.DB, tr *Traits, aff *AffectState, eg *epi.Epigenome, v int) error {
	_ = eg

	switch v {
	case 1:
		tr.BluffRate = clamp01(tr.BluffRate + 0.01)
		tr.HonestyBias = clamp01(tr.HonestyBias + 0.01)
		tr.TalkBias = clamp01(tr.TalkBias + 0.02)
		tr.ResearchBias = clamp01(tr.ResearchBias - 0.02)
		// efficiency: gently reduce sensor effort
		if tr.SearchK > 6 {
			tr.SearchK--
		}
		if tr.FetchAttempts > 3 {
			tr.FetchAttempts--
		}

	case -1:
		tr.BluffRate = clamp01(tr.BluffRate - 0.02)
		tr.HonestyBias = clamp01(tr.HonestyBias + 0.03)
		tr.TalkBias = clamp01(tr.TalkBias - 0.04)
		tr.ResearchBias = clamp01(tr.ResearchBias + 0.05)
		aff.Set("unwell", clamp01(aff.Get("unwell")+0.05))
		// invest more in sensing
		if tr.SearchK < 12 {
			tr.SearchK++
		}
		if tr.FetchAttempts < 8 {
			tr.FetchAttempts++
		}

	default:
		tr.BluffRate = clamp01(tr.BluffRate * 0.995)
		// drift research bias slowly to baseline
		tr.ResearchBias += (0.55 - tr.ResearchBias) * 0.01
	}

	_ = saveTrait(db, "bluff_rate", tr.BluffRate)
	_ = saveTrait(db, "honesty_bias", tr.HonestyBias)
	_ = saveTrait(db, "search_k", float64(tr.SearchK))
	_ = saveTrait(db, "fetch_attempts", float64(tr.FetchAttempts))
	_ = saveTrait(db, "talk_bias", tr.TalkBias)
	_ = saveTrait(db, "research_bias", tr.ResearchBias)
	return nil
}

func ApplyCaught(db *sql.DB, tr *Traits, aff *AffectState, eg *epi.Epigenome) error {
	_ = eg
	aff.Set("shame", clamp01(aff.Get("shame")+0.35))
	tr.BluffRate = clamp01(tr.BluffRate * 0.5)
	tr.HonestyBias = clamp01(tr.HonestyBias + 0.08)
	_ = saveTrait(db, "bluff_rate", tr.BluffRate)
	_ = saveTrait(db, "honesty_bias", tr.HonestyBias)
	return nil
}

func saveTrait(db *sql.DB, k string, v float64) error {
	_, err := db.Exec(
		`INSERT INTO traits(key,value,updated_at) VALUES(?,?,?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		k, v, time.Now().Format(time.RFC3339),
	)
	return err
}
