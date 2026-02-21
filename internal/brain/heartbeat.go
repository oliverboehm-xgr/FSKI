package brain

import (
	"time"

	"frankenstein-v0/internal/epi"
)

type Heartbeat struct {
	eg *epi.Epigenome
}

func NewHeartbeat(eg *epi.Epigenome) *Heartbeat {
	return &Heartbeat{eg: eg}
}

// Start runs a background ticker. The callback must be fast and lock-protected by caller.
func (h *Heartbeat) Start(onTick func(delta time.Duration)) (stop func()) {
	interval := h.eg.HeartbeatInterval()
	if interval <= 0 {
		interval = 500 * time.Millisecond
	}
	t := time.NewTicker(interval)
	done := make(chan struct{})

	go func() {
		last := time.Now()
		for {
			select {
			case <-t.C:
				now := time.Now()
				delta := now.Sub(last)
				last = now
				onTick(delta)
			case <-done:
				t.Stop()
				return
			}
		}
	}()

	return func() { close(done) }
}
