package main

import "testing"

func TestExtractNumberedListItem(t *testing.T) {
	text := `Hier sind Punkte:
1. Eins
2. Zwei
**3.** Drei fett
4. Vier normal`

	if got := ExtractNumberedListItem(text, 2); got != "Zwei" {
		t.Fatalf("expected 'Zwei', got %q", got)
	}
	if got := ExtractNumberedListItem(text, 3); got != "Drei fett" {
		t.Fatalf("expected 'Drei fett', got %q", got)
	}
	if got := ExtractNumberedListItem(text, 5); got != "" {
		t.Fatalf("expected empty for missing item, got %q", got)
	}
}
