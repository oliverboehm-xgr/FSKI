package codeindex

import (
	"database/sql"
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"time"
)

type Symbols struct {
	Funcs []string `json:"funcs"`
	Types []string `json:"types"`
	Vars  []string `json:"vars"`
}

func IndexRepo(db *sql.DB, root string) error {
	if db == nil {
		return nil
	}
	root = filepath.Clean(root)
	now := time.Now().Format(time.RFC3339)

	return filepath.WalkDir(root, func(path string, d os.DirEntry, err error) error {
		if err != nil || d == nil {
			return nil
		}
		if d.IsDir() {
			// skip vendor/.git
			base := filepath.Base(path)
			if base == ".git" || base == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}
		if !strings.HasSuffix(path, ".go") {
			return nil
		}
		// ignore generated testdata if you like
		rel, _ := filepath.Rel(root, path)
		fset := token.NewFileSet()
		f, perr := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if perr != nil || f == nil {
			return nil
		}

		pkg := f.Name.Name
		sy := Symbols{}
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.FuncDecl:
				if x.Name != nil {
					sy.Funcs = append(sy.Funcs, x.Name.Name)
				}
			case *ast.TypeSpec:
				if x.Name != nil {
					sy.Types = append(sy.Types, x.Name.Name)
				}
			case *ast.ValueSpec:
				for _, n := range x.Names {
					if n != nil {
						sy.Vars = append(sy.Vars, n.Name)
					}
				}
			}
			return true
		})

		// lightweight summary (deterministic): package + top symbols
		sum := "package " + pkg
		if len(sy.Types) > 0 {
			sum += "; types: " + strings.Join(trimList(sy.Types, 10), ", ")
		}
		if len(sy.Funcs) > 0 {
			sum += "; funcs: " + strings.Join(trimList(sy.Funcs, 12), ", ")
		}

		js, _ := json.Marshal(sy)
		_, _ = db.Exec(
			`INSERT INTO code_index(path,package,summary,symbols_json,updated_at)
			 VALUES(?,?,?,?,?)
			 ON CONFLICT(path) DO UPDATE SET package=excluded.package, summary=excluded.summary, symbols_json=excluded.symbols_json, updated_at=excluded.updated_at`,
			rel, pkg, sum, string(js), now,
		)
		return nil
	})
}

func trimList(in []string, n int) []string {
	if len(in) <= n {
		return in
	}
	return in[:n]
}
