package brain

import (
	"database/sql"
	"encoding/json"
	"strings"
	"time"

	"frankenstein-v0/internal/epi"
	"frankenstein-v0/internal/ollama"
)

// BootstrapEpigenomeEvolution creates epigenome_proposals that help Bunny self-heal common deployment issues
// (missing area models, etc.). It does NOT auto-apply; the user can inspect/apply via /epi.
func BootstrapEpigenomeEvolution(db *sql.DB, oc *ollama.Client, eg *epi.Epigenome) {
	if db == nil || eg == nil || oc == nil {
		return
	}

	// throttle (avoid spamming proposals on restart loops)
	var last string
	_ = db.QueryRow(`SELECT created_at FROM epigenome_proposals ORDER BY id DESC LIMIT 1`).Scan(&last)
	if ts, err := time.Parse(time.RFC3339, strings.TrimSpace(last)); err == nil {
		if time.Since(ts) < 5*time.Minute {
			return
		}
	}

	installed, err := oc.ListModels()
	if err != nil || len(installed) == 0 {
		return
	}

	speaker := strings.TrimSpace(eg.ModelFor("speaker", ""))
	if speaker == "" {
		speaker = strings.TrimSpace(eg.ModelFor("default", "llama3.1:8b"))
	}

	checkArea := func(area string) {
		want := strings.TrimSpace(eg.ModelFor(area, speaker))
		if want == "" {
			return
		}
		if _, ok := installed[want]; ok {
			return
		}
		patch := map[string]any{"modules": map[string]any{"models": map[string]any{"params": map[string]any{area: speaker}}}}
		b, _ := json.Marshal(patch)
		note := "configured model missing: " + area + "=" + want + "; fallback to speaker=" + speaker
		_, _ = InsertEpigenomeProposal(db, "models.fallback."+area, string(b), note)
	}

	checkArea("scout")
	checkArea("critic")
	checkArea("hippocampus")
}
