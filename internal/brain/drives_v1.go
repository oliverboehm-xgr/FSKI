package brain

import (
	"database/sql"
	"encoding/json"
	"math"
	"time"

	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/sensors"
)

type DrivesV1 struct {
	Energy       float64
	Survival     float64
	Curiosity    float64
	UserImprove  float64
	SocSat       float64
	UrgeInteract float64

	UserRewardEMA float64
	CaughtEMA     float64
	LastHelpAt    time.Time
}

type ResourceMetrics struct {
	DiskFreeBytes  uint64  `json:"disk_free_bytes"`
	DiskTotalBytes uint64  `json:"disk_total_bytes"`
	RamFreeBytes   uint64  `json:"ram_free_bytes"`
	RamTotalBytes  uint64  `json:"ram_total_bytes"`
	CPUUtil        float64 `json:"cpu_util"`
	LatencyEMAms   float64 `json:"latency_ema_ms"`
}

func dangerExp(r float64, k float64) float64 {
	r = clamp01(r)
	if k <= 0 {
		k = 4
	}
	return math.Exp(-k * r)
}

func energyFromResources(p epi.DrivesV1Params, rm ResourceMetrics, latencyEMAms float64) (energy float64, rDisk, rRam, rCPU, rLat float64) {
	if p.DiskTargetBytes <= 0 {
		p.DiskTargetBytes = 1e10
	}
	if p.RamTargetBytes <= 0 {
		p.RamTargetBytes = 3e9
	}
	rDisk = clamp01(float64(rm.DiskFreeBytes) / p.DiskTargetBytes)
	rRam = clamp01(float64(rm.RamFreeBytes) / p.RamTargetBytes)
	rCPU = clamp01(1.0 - clamp01(rm.CPUUtil))
	lt := latencyEMAms
	if lt < 0 {
		lt = 0
	}
	if p.LatencyTargetMs <= 50 {
		p.LatencyTargetMs = 2500
	}
	rLat = math.Exp(-lt / p.LatencyTargetMs)
	ws := p.Wdisk + p.Wram + p.Wcpu + p.Wlat + p.Werr
	if ws <= 0 {
		ws = 1
	}
	wDisk := p.Wdisk / ws
	wRam := p.Wram / ws
	wCPU := p.Wcpu / ws
	wLat := p.Wlat / ws
	wErr := p.Werr / ws
	rErr := 1.0
	energy = wDisk*rDisk + wRam*rRam + wCPU*rCPU + wLat*rLat + wErr*rErr
	return clamp01(energy), rDisk, rRam, rCPU, rLat
}

func UpdateResources(db *sql.DB, path string, snap sensors.Snapshot, latencyEMAms float64) (ResourceMetrics, error) {
	rm := ResourceMetrics{DiskFreeBytes: snap.DiskFreeBytes, DiskTotalBytes: snap.DiskTotalBytes, RamFreeBytes: snap.RamFreeBytes, RamTotalBytes: snap.RamTotalBytes, CPUUtil: snap.CPUUtil, LatencyEMAms: latencyEMAms}
	if db == nil {
		return rm, nil
	}
	now := time.Now().Format(time.RFC3339)
	rid := "disk:" + path
	metrics, _ := json.Marshal(rm)
	_, _ = db.Exec(`INSERT INTO resources(id,kind,present,metrics_json,constraints_json,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET present=excluded.present, metrics_json=excluded.metrics_json, updated_at=excluded.updated_at`, rid, "capacity", 1, string(metrics), "{}", now)
	_, _ = db.Exec(`INSERT INTO resources(id,kind,present,metrics_json,constraints_json,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET present=excluded.present, metrics_json=excluded.metrics_json, updated_at=excluded.updated_at`, "ram", "capacity", 1, string(metrics), "{}", now)
	_, _ = db.Exec(`INSERT INTO resources(id,kind,present,metrics_json,constraints_json,updated_at) VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET present=excluded.present, metrics_json=excluded.metrics_json, updated_at=excluded.updated_at`, "cpu", "capacity", 1, string(metrics), "{}", now)
	return rm, nil
}

func lastUserMessageAt(db *sql.DB) time.Time {
	if db == nil {
		return time.Time{}
	}
	var ts string
	_ = db.QueryRow(`SELECT m.created_at FROM messages m JOIN message_meta mm ON mm.message_id=m.id WHERE mm.kind='user' ORDER BY m.id DESC LIMIT 1`).Scan(&ts)
	t, _ := time.Parse(time.RFC3339, ts)
	return t
}

func computeUserRewardEMA(db *sql.DB, alpha float64) (reward float64, caught float64) {
	if db == nil {
		return 0, 0
	}
	if alpha <= 0 || alpha > 1 {
		alpha = 0.12
	}
	rows, err := db.Query(`SELECT value FROM ratings ORDER BY created_at DESC LIMIT 50`)
	if err == nil {
		defer rows.Close()
		ema := 0.0
		init := false
		for rows.Next() {
			var v int
			_ = rows.Scan(&v)
			x := float64(v)
			if x > 1 {
				x = 1
			}
			if x < -1 {
				x = -1
			}
			if !init {
				ema = x
				init = true
			} else {
				ema = (1-alpha)*ema + alpha*x
			}
		}
		reward = ema
	}
	var n int
	_ = db.QueryRow(
		`SELECT COUNT(*) FROM caught_events WHERE created_at >= ?`,
		time.Now().Add(-60*time.Minute).Format(time.RFC3339),
	).Scan(&n)
	caught = 1.0 - math.Exp(-0.5*float64(n))
	if caught < 0 {
		caught = 0
	}
	if caught > 1 {
		caught = 1
	}
	return reward, caught
}

func TickDrivesV1(db *sql.DB, eg *epi.Epigenome, d *DrivesV1, ws *Workspace, aff *AffectState, snap sensors.Snapshot, latencyEMAms float64, activeTopic string, conceptConf float64, stanceConf float64) {
	_ = ws
	_ = activeTopic
	if eg == nil || d == nil || aff == nil {
		return
	}
	p := eg.DrivesV1()
	if !p.Enabled {
		return
	}
	rm, _ := UpdateResources(db, p.DiskPath, snap, latencyEMAms)
	energy, rDisk, rRam, _, rLat := energyFromResources(p, rm, latencyEMAms)
	d.Energy = energy
	gDisk := dangerExp(rDisk, p.Kdisk)
	gRam := dangerExp(rRam, p.Kram)
	gCPU := dangerExp(1.0-clamp01(rm.CPUUtil), p.Kcpu)
	gLat := clamp01(1.0 - rLat)
	wsum := p.Wdisk + p.Wram + p.Wcpu + p.Wlat
	if wsum <= 0 {
		wsum = 1
	}
	Dsurv := (p.Wdisk/wsum)*gDisk + (p.Wram/wsum)*gRam + (p.Wcpu/wsum)*gCPU + (p.Wlat/wsum)*gLat
	d.Survival = clamp01(Dsurv)
	aff.Ensure("pain", 0.0)
	aff.Ensure("anxiety", 0.0)
	pain := aff.Get("pain")
	anx := aff.Get("anxiety")
	kgap := 1.0 - math.Max(conceptConf, stanceConf)
	if kgap < 0 {
		kgap = 0
	}
	if kgap > 1 {
		kgap = 1
	}
	pain = clamp01(pain + 0.10*(d.Survival*d.Survival) - 0.015)
	anx = clamp01(anx + 0.06*(d.Survival*(0.5+0.5*kgap)) - 0.012)
	aff.Set("pain", pain)
	aff.Set("anxiety", anx)
	lastU := lastUserMessageAt(db)
	if lastU.IsZero() {
		lastU = time.Now().Add(-24 * time.Hour)
	}
	dt := time.Since(lastU).Seconds()
	tau := p.TauSocialSec
	if tau < 60 {
		tau = 60
	}
	d.SocSat = math.Exp(-dt / tau)
	craving := clamp01(1.0 - d.SocSat)
	reward, caught := computeUserRewardEMA(db, p.EmaUser)
	d.UserRewardEMA = reward
	d.CaughtEMA = clamp01((1-p.EmaCaught)*d.CaughtEMA + p.EmaCaught*caught)
	d.UserImprove = clamp01(0.40 + 0.80*math.Max(0, -d.UserRewardEMA) + 1.00*d.CaughtEMA)
	d.Curiosity = clamp01(0.45 + 0.80*kgap - 0.60*d.Survival)
	aff.Ensure("shame", 0.0)
	sh := aff.Get("shame")
	d.UrgeInteract = clamp01(0.30 + 0.90*craving - 0.50*sh - 0.70*d.Survival)
	aff.Ensure("satisfaction", 0.0)
	learnSat := clamp01(1.0 - kgap)
	userPos := clamp01((d.UserRewardEMA + 1.0) / 2.0)
	satTarget := clamp01(0.50*d.SocSat + 0.30*learnSat + 0.20*userPos)
	sat := aff.Get("satisfaction")
	sat = clamp01(sat + 0.08*(satTarget-sat))
	aff.Set("satisfaction", sat)
}
