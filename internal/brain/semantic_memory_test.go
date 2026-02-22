package brain

import (
	"path/filepath"
	"testing"

	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/state"
)

func TestSemanticMemoryStep_StoresIdentityClaim(t *testing.T) {
	db, err := state.Open(filepath.Join(t.TempDir(), "brain.sqlite"))
	if err != nil {
		t.Fatalf("init db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	eg, err := epi.LoadOrInit(filepath.Join(t.TempDir(), "epi.json"))
	if err != nil {
		t.Fatalf("load epigenome: %v", err)
	}

	handled, reply := SemanticMemoryStep(db.DB, eg, "Dr. Oliver Böhm das bin übrigens ich")
	if !handled {
		t.Fatalf("expected semantic memory to handle identity claim")
	}
	if reply == "" {
		t.Fatalf("expected ack reply for identity claim")
	}

	got, ok := GetFact(db.DB, "user", "self_identity")
	if !ok {
		t.Fatalf("expected self_identity fact to be persisted")
	}
	if got != "Dr. Oliver Böhm" {
		t.Fatalf("unexpected identity object: %q", got)
	}
}
