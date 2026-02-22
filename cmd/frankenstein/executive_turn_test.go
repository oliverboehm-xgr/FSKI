package main

import (
	"strings"
	"testing"
)

func TestNormalizeUnifiedDiffHunks_RepairsEmptyHunkLine(t *testing.T) {
	diff := strings.Join([]string{
		"diff --git a/a.go b/a.go",
		"--- a/a.go",
		"+++ b/a.go",
		"@@ -1,3 +1,3 @@",
		" package main",
		"",
		"-func a() {}",
		"+func a() { /*x*/ }",
	}, "\n")

	norm := normalizeUnifiedDiffHunks(diff)
	if strings.Contains(norm, "\n\n-func") {
		t.Fatalf("expected empty hunk line to be normalized, got:\n%s", norm)
	}
	if err := validateUnifiedDiffSyntax(norm); err != nil {
		t.Fatalf("expected normalized diff to be valid, got err: %v", err)
	}
}
