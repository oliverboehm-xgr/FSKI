package brain

import (
	"database/sql"
	"time"
)

// Drives are the "central brain" motivational state. Keep it simple and generic.
// Values are 0..1.
type Drives struct {
	Curiosity   float64
	UrgeToShare float64
}

func LoadOrInitDrives(db *sql.DB) (*Drives, error) {
	d := &Drives{Curiosity: 0.45, UrgeToShare: 0.20}
	if db == nil {
		return d, nil
	}
	rows, err := db.Query(`SELECT key, value FROM drive_state`)
	if err != nil {
		_ = saveDrive(db, "curiosity", d.Curiosity)
		_ = saveDrive(db, "urge_to_share", d.UrgeToShare)
		return d, nil
	}
	defer rows.Close()
	for rows.Next() {
		var k string
		var v float64
		_ = rows.Scan(&k, &v)
		switch k {
		case "curiosity":
			d.Curiosity = clamp01(v)
		case "urge_to_share":
			d.UrgeToShare = clamp01(v)
		}
	}
	_ = saveDrive(db, "curiosity", d.Curiosity)
	_ = saveDrive(db, "urge_to_share", d.UrgeToShare)
	return d, nil
}

func SaveDrives(db *sql.DB, d *Drives) {
	if db == nil || d == nil {
		return
	}
	_ = saveDrive(db, "curiosity", clamp01(d.Curiosity))
	_ = saveDrive(db, "urge_to_share", clamp01(d.UrgeToShare))
}

func saveDrive(db *sql.DB, k string, v float64) error {
	_, err := db.Exec(
		`INSERT INTO drive_state(key,value,updated_at) VALUES(?,?,?)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at`,
		k, v, time.Now().Format(time.RFC3339),
	)
	return err
}

// TickDrives: generic homeostasis + coupling to affects (pain/fear/shame reduce urge)
func TickDrives(d *Drives, aff *AffectState, dt time.Duration) {
	if d == nil {
		return
	}
	sec := dt.Seconds()
	if sec <= 0 {
		return
	}

	// baseline relaxation
	d.Curiosity += (0.45 - d.Curiosity) * clamp01(0.02*sec)
	d.UrgeToShare += (0.20 - d.UrgeToShare) * clamp01(0.03*sec)

	if aff != nil {
		// inhibit urge when negative states are high
		inhib := 0.0
		inhib += 0.7 * aff.Get("shame")
		inhib += 0.4 * aff.Get("fear")
		inhib += 0.3 * aff.Get("pain")
		inhib += 0.2 * aff.Get("unwell")
		d.UrgeToShare = clamp01(d.UrgeToShare - inhib*0.05*sec)

		// shame also slightly inhibits curiosity (self-check mode)
		d.Curiosity = clamp01(d.Curiosity - aff.Get("shame")*0.02*sec)
	}
}
