package main

import (
	"path/filepath"
	"strings"
	"testing"

	"frankenstein-v0/internal/state"
)

func TestBuildReferenceCandidates_PicksRecentSemanticallyMatchingTurns(t *testing.T) {
	db, err := state.Open(filepath.Join(t.TempDir(), "ctx.sqlite"))
	if err != nil {
		t.Fatalf("open db: %v", err)
	}
	t.Cleanup(func() { _ = db.Close() })

	_, _ = db.Exec(`INSERT INTO messages(id,created_at,priority,text,sources_json) VALUES(100,'t',0.5,'1. Foo 2. Iran Schlagzeile 3. Bar','[]')`)
	_, _ = db.Exec(`INSERT INTO message_meta(message_id,kind) VALUES(100,'reply')`)
	_, _ = db.Exec(`INSERT INTO messages(id,created_at,priority,text,sources_json) VALUES(101,'t',0.5,'Lass uns über Iran weiterreden','[]')`)
	_, _ = db.Exec(`INSERT INTO message_meta(message_id,kind) VALUES(101,'user')`)
	_, _ = db.Exec(`INSERT INTO messages(id,created_at,priority,text,sources_json) VALUES(102,'t',0.5,'Börse und DAX heute','[]')`)
	_, _ = db.Exec(`INSERT INTO message_meta(message_id,kind) VALUES(102,'reply')`)

	out := BuildReferenceCandidates(db.DB, "lass uns über den iran punkt sprechen", 2)
	if !strings.Contains(out, "Iran") {
		t.Fatalf("expected Iran candidate in reference anchors, got: %q", out)
	}
}
