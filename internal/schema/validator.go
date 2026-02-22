package schema

import (
	"errors"
	"strings"
)

// ValidateSchemaSQL performs conservative validation for schema changes.
// Allowed: CREATE TABLE, CREATE INDEX, ALTER TABLE ... ADD COLUMN
// Disallowed: DROP, DELETE, UPDATE, INSERT, PRAGMA, ATTACH, VACUUM, TRIGGER, VIEW
func ValidateSchemaSQL(sqlText string) error {
	s := strings.TrimSpace(sqlText)
	if s == "" {
		return errors.New("empty sql")
	}
	ls := strings.ToLower(s)
	bad := []string{"drop ", "delete ", "update ", "insert ", "pragma ", "attach ", "vacuum", "trigger", "view", "replace ", "alter table", "begin", "commit"}
	for _, b := range bad {
		if strings.Contains(ls, b) {
			// allow ALTER TABLE only for ADD COLUMN; checked below
			if b == "alter table" {
				continue
			}
			return errors.New("disallowed sql keyword: " + b)
		}
	}

	parts := strings.Split(s, ";")
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		lp := strings.ToLower(p)
		if strings.HasPrefix(lp, "create table ") || strings.HasPrefix(lp, "create index ") || strings.HasPrefix(lp, "create unique index ") {
			continue
		}
		if strings.HasPrefix(lp, "alter table ") {
			if !strings.Contains(lp, " add column ") {
				return errors.New("only ALTER TABLE ... ADD COLUMN allowed")
			}
			continue
		}
		return errors.New("statement not allowed: " + firstWord(lp))
	}
	return nil
}

func firstWord(s string) string {
	s = strings.TrimSpace(s)
	if s == "" {
		return ""
	}
	if i := strings.IndexByte(s, ' '); i > 0 {
		return s[:i]
	}
	return s
}
