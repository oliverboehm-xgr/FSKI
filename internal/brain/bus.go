package brain

import (
	"database/sql"
	"time"

	"frankenstein-v0/internal/epi"
)

// Action is a cortex-bus output. It is NOT text. It is an intention.
type Action interface {
	Kind() string
	Priority() float64
}

type ActionDaydream struct{ P float64 }

func (a ActionDaydream) Kind() string      { return "daydream" }
func (a ActionDaydream) Priority() float64 { return a.P }

type ActionSpeak struct {
	P      float64
	Reason string
	Topic  string
}

func (a ActionSpeak) Kind() string      { return "speak" }
func (a ActionSpeak) Priority() float64 { return a.P }

type TickContext struct {
	DB    *sql.DB
	EG    *epi.Epigenome
	Body  any // leave generic; main owns concrete types
	WS    *Workspace
	Aff   *AffectState
	Tr    any
	Dr    *Drives
	Now   time.Time
	Delta time.Duration
}

type Area interface {
	Name() string
	Tick(ctx *TickContext) []Action
}

type Bus struct {
	areas []Area
}

func NewBus(areas ...Area) *Bus {
	return &Bus{areas: areas}
}

func (b *Bus) Tick(ctx *TickContext) []Action {
	if b == nil || ctx == nil {
		return nil
	}
	var out []Action
	for _, a := range b.areas {
		acts := a.Tick(ctx)
		if len(acts) > 0 {
			out = append(out, acts...)
		}
	}
	return out
}
