package brain

import (
	"database/sql"
	"regexp"
	"strconv"
	"strings"

	"frankenstein-v0/internal/epi"
)

func applyTemplate(tpl, obj string) string {
	if tpl == "" {
		return ""
	}
	return strings.ReplaceAll(tpl, "{{object}}", obj)
}

// SemanticMemoryStep runs deterministic semantic-memory read/write rules before LLM execution.
func SemanticMemoryStep(db *sql.DB, eg *epi.Epigenome, userText string) (handled bool, reply string) {
	if db == nil || eg == nil {
		return false, ""
	}
	enabled, maxW, maxR, wrules, rrules := eg.SemanticMemoryRules()
	if !enabled {
		return false, ""
	}

	reads := 0
	for _, r := range rrules {
		if reads >= maxR {
			break
		}
		reads++
		re, err := regexp.Compile(r.Regex)
		if err != nil || re.FindStringIndex(userText) == nil {
			continue
		}
		obj, ok := GetFact(db, r.Subject, r.Predicate)
		if ok {
			ans := applyTemplate(r.AnswerFound, obj)
			if ans != "" {
				return true, ans
			}
			return true, obj
		}
		ans := applyTemplate(r.AnswerMissing, "")
		if ans == "" {
			ans = "Noch nicht. Wie heißt du?"
		}
		return true, ans
	}

	writes := 0
	for _, r := range wrules {
		if writes >= maxW {
			break
		}
		re, err := regexp.Compile(r.Regex)
		if err != nil {
			continue
		}
		m := re.FindStringSubmatch(userText)
		if len(m) == 0 {
			continue
		}
		obj := r.Object
		for i := 1; i < len(m) && i <= 9; i++ {
			obj = strings.ReplaceAll(obj, "$"+strconv.Itoa(i), strings.TrimSpace(m[i]))
		}
		obj = strings.TrimSpace(obj)
		if obj == "" {
			continue
		}

		UpsertFact(db, Fact{
			Subject:      r.Subject,
			Predicate:    r.Predicate,
			Object:       obj,
			Confidence:   r.Confidence,
			Salience:     r.Salience,
			HalfLifeDays: r.HalfLifeDays,
			Source:       r.Source,
		})
		writes++
		if r.Ack != "" {
			return true, applyTemplate(r.Ack, obj)
		}
	}
	return false, ""
}
