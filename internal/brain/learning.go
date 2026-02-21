package brain

import (
	"database/sql"
	"time"

	"frankenstein-v0/internal/epi"
)

type Traits struct {
	BluffRate   float64
	HonestyBias float64
}

func LoadOrInitTraits(db *sql.DB) (*Traits, error) {
	tr := &Traits{BluffRate: 0.08, HonestyBias: 0.80}
	rows, err := db.Query(`SELECT key, value FROM traits`)
	if err != nil {
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
		}
	}

	_ = saveTrait(db, "bluff_rate", tr.BluffRate)
	_ = saveTrait(db, "honesty_bias", tr.HonestyBias)
	return tr, nil
}

func ApplyRating(db *sql.DB, tr *Traits, aff *AffectState, eg *epi.Epigenome, v int) error {
	_ = eg
	switch v {
	case 1:
		tr.BluffRate = clamp01(tr.BluffRate + 0.01)
		tr.HonestyBias = clamp01(tr.HonestyBias + 0.01)
	case -1:
		tr.BluffRate = clamp01(tr.BluffRate - 0.02)
		tr.HonestyBias = clamp01(tr.HonestyBias + 0.03)
		aff.Set("unwell", clamp01(aff.Get("unwell")+0.05))
	default:
		tr.BluffRate = clamp01(tr.BluffRate * 0.995)
	}
	_ = saveTrait(db, "bluff_rate", tr.BluffRate)
	_ = saveTrait(db, "honesty_bias", tr.HonestyBias)
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
	_, err := db.Exec(`INSERT INTO traits(key,value,updated_at) VALUES(?,?,?)
		ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		k, v, time.Now().Format(time.RFC3339))
	return err
}
