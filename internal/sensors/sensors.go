package sensors

import "time"

type Snapshot struct {
	DiskFreeBytes  uint64
	DiskTotalBytes uint64
	RamFreeBytes   uint64
	RamTotalBytes  uint64
	CPUUtil        float64
}

type Sampler interface {
	Sample(path string) (Snapshot, error)
}

type Latency struct {
	EMAms float64
}

type Clock interface {
	Now() time.Time
}
