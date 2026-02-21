package ollama

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

type EnsureResult struct {
	Available bool
	Started   bool
	Pulled    []string
	Missing   []string
	Err       error
}

func EnsureAvailable(ctx context.Context, c *Client, wantModels []string, autoStart, autoPull bool, startRetries int, startRetry time.Duration, pullTimeout time.Duration, maxPull int) EnsureResult {
	res := EnsureResult{}
	if c == nil {
		res.Err = errors.New("nil ollama client")
		return res
	}

	if err := c.Ping(); err != nil {
		if autoStart {
			if started := tryStartOllama(); started {
				res.Started = true
			}
			for i := 0; i < startRetries; i++ {
				select {
				case <-ctx.Done():
					res.Err = ctx.Err()
					return res
				default:
				}
				time.Sleep(startRetry)
				if err2 := c.Ping(); err2 == nil {
					break
				}
			}
		}
	}

	if err := c.Ping(); err != nil {
		res.Available = false
		res.Err = err
		return res
	}
	res.Available = true

	models, err := c.ListModels()
	if err != nil {
		res.Err = err
		return res
	}

	want := uniqNonEmpty(wantModels)
	var missing []string
	for _, m := range want {
		if _, ok := models[m]; !ok {
			missing = append(missing, m)
		}
	}
	res.Missing = missing
	if len(missing) == 0 || !autoPull || maxPull == 0 {
		return res
	}

	toPull := missing
	if len(toPull) > maxPull {
		toPull = toPull[:maxPull]
	}
	for _, m := range toPull {
		if err := pullModel(ctx, m, pullTimeout); err == nil {
			res.Pulled = append(res.Pulled, m)
		}
	}

	models2, _ := c.ListModels()
	var still []string
	for _, m := range missing {
		if _, ok := models2[m]; !ok {
			still = append(still, m)
		}
	}
	res.Missing = still
	return res
}

func uniqNonEmpty(in []string) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, s := range in {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func tryStartOllama() bool {
	if runtime.GOOS == "windows" {
		_ = exec.Command("sc.exe", "start", "Ollama").Run()
	}
	cmd := exec.Command("ollama", "serve")
	_ = cmd.Start()
	return true
}

func pullModel(ctx context.Context, model string, timeout time.Duration) error {
	cctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	cmd := exec.CommandContext(cctx, "ollama", "pull", model)
	return cmd.Run()
}

func FormatEnsure(res EnsureResult) string {
	if !res.Available {
		return fmt.Sprintf("LLM backend OFFLINE (%v)", res.Err)
	}
	s := "LLM backend ONLINE"
	if res.Started {
		s += " (started)"
	}
	if len(res.Pulled) > 0 {
		s += " pulled: " + strings.Join(res.Pulled, ", ")
	}
	if len(res.Missing) > 0 {
		s += " missing: " + strings.Join(res.Missing, ", ")
	}
	return s
}
