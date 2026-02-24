package websense

import (
	"errors"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

type SpiderBudget struct {
	MaxPages      int
	MaxBytesTotal int64
	PerDomainMax  int
	Timeout       time.Duration
	MaxLinksPerPage int
}

// Spider crawls starting from seed URLs, following href links with a simple BFS.
// v1 goal: provide a quality building block for "iterative / recursive websense" with budget constraints.
func Spider(seeds []string, bud SpiderBudget) ([]*FetchResult, error) {
	if len(seeds) == 0 {
		return nil, errors.New("no seeds")
	}
	if bud.MaxPages <= 0 {
		bud.MaxPages = 6
	}
	if bud.MaxBytesTotal <= 0 {
		bud.MaxBytesTotal = 5_000_000
	}
	if bud.PerDomainMax <= 0 {
		bud.PerDomainMax = 3
	}
	if bud.Timeout <= 0 {
		bud.Timeout = 12 * time.Second
	}
	if bud.MaxLinksPerPage <= 0 {
		bud.MaxLinksPerPage = 12
	}

	seen := map[string]bool{}
	dCount := map[string]int{}
	queue := make([]string, 0, len(seeds))
	for _, s := range seeds {
		u := normalizeResultURL(s)
		if u == "" || seen[u] { continue }
		seen[u] = true
		queue = append(queue, u)
	}

	client := &http.Client{Timeout: bud.Timeout}
	var out []*FetchResult
	var used int64

	for len(queue) > 0 && len(out) < bud.MaxPages && used < bud.MaxBytesTotal {
		u := queue[0]
		queue = queue[1:]

		pu, err := url.Parse(u)
		if err != nil || pu.Hostname() == "" {
			continue
		}
		dom := strings.ToLower(pu.Hostname())
		if dCount[dom] >= bud.PerDomainMax {
			continue
		}

		req, _ := http.NewRequest("GET", u, nil)
		applyDefaultHeaders(req)
		resp, err := client.Do(req)
		if err != nil {
			continue
		}
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1_500_000))
		resp.Body.Close()
		used += int64(len(b))
		if used >= bud.MaxBytesTotal {
			break
		}

		page := string(b)
		ct := strings.ToLower(resp.Header.Get("Content-Type"))
		if strings.Contains(ct, "text/html") || ct == "" {
			links := extractLinks(page, u, bud.MaxLinksPerPage)
			for _, lk := range links {
				n := normalizeResultURL(lk)
				if n == "" || seen[n] {
					continue
				}
				seen[n] = true
				queue = append(queue, n)
			}
		}

		// reuse existing cleaner (stripHTML + normalizeWhitespace) via Fetch-style logic
		txt := normalizeWhitespace(stripHTML(page))
		fr := &FetchResult{
			Title:     extractTitle(page),
			URL:       u,
			Text:      txt,
			Snippet:   func() string { if len(txt) > 420 { return txt[:420] }; return txt }(),
			Body:      func() string { if len(txt) > 3000 { return txt[:3000] }; return txt }(),
			FetchedAt: time.Now(),
			Domain:    dom,
		}
		out = append(out, fr)
		dCount[dom]++
	}

	return out, nil
}

func extractLinks(htmlPage string, base string, max int) []string {
	if max <= 0 {
		max = 12
	}
	re := regexp.MustCompile(`(?is)href=["']([^"'#]+)["']`)
	m := re.FindAllStringSubmatch(htmlPage, max)
	out := make([]string, 0, len(m))
	baseU, _ := url.Parse(base)
	for _, mm := range m {
		if len(mm) < 2 {
			continue
		}
		h := strings.TrimSpace(mm[1])
		if h == "" {
			continue
		}
		u, err := url.Parse(h)
		if err != nil {
			continue
		}
		if baseU != nil {
			u = baseU.ResolveReference(u)
		}
		out = append(out, u.String())
	}
	return out
}
