package websense

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

type SearchResult struct {
	Title   string
	URL     string
	Snippet string
}

type FetchResult struct {
	Title     string
	URL       string
	Text      string
	Snippet   string
	Body      string // first 3000 chars for LLM context
	Hash      string
	FetchedAt time.Time
	Domain    string
}

var httpClient = &http.Client{
	Timeout: 12 * time.Second,
}

// DuckDuckGo HTML (v0, aber: robustere Links + Snippets).
func Search(query string, k int) ([]SearchResult, error) {
	if k <= 0 {
		k = 6
	}
	q := url.QueryEscape(query)
	u := "https://duckduckgo.com/html/?q=" + q

	req, _ := http.NewRequest("GET", u, nil)
	applyDefaultHeaders(req)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, errors.New("search http status: " + resp.Status)
	}

	b, _ := io.ReadAll(io.LimitReader(resp.Body, 2_000_000))
	page := string(b)

	// Titles/URLs
	reA := regexp.MustCompile(`(?is)<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>`)
	mA := reA.FindAllStringSubmatch(page, k)

	// Snippets (DDG nutzt je nach Variante <a> oder <div>)
	var snippets []string
	reSnipA := regexp.MustCompile(`(?is)<a[^>]*class="result__snippet"[^>]*>(.*?)</a>`)
	for _, mm := range reSnipA.FindAllStringSubmatch(page, k) {
		snippets = append(snippets, normalizeWhitespace(stripHTML(mm[1])))
	}
	if len(snippets) == 0 {
		reSnipD := regexp.MustCompile(`(?is)<div[^>]*class="result__snippet"[^>]*>(.*?)</div>`)
		for _, mm := range reSnipD.FindAllStringSubmatch(page, k) {
			snippets = append(snippets, normalizeWhitespace(stripHTML(mm[1])))
		}
	}

	out := make([]SearchResult, 0, len(mA))
	for i, mm := range mA {
		raw := html.UnescapeString(mm[1])
		link := normalizeResultURL(raw)
		title := normalizeWhitespace(stripHTML(mm[2]))

		snip := ""
		if i < len(snippets) {
			snip = snippets[i]
		}
		out = append(out, SearchResult{
			Title:   title,
			URL:     link,
			Snippet: snip,
		})
	}
	return out, nil
}

func Fetch(rawURL string) (*FetchResult, error) {
	normalized := normalizeResultURL(strings.TrimSpace(rawURL))
	pu, err := url.Parse(normalized)
	if err != nil {
		return nil, err
	}
	if pu.Scheme == "" {
		return nil, errors.New("fetch: missing scheme")
	}

	req, _ := http.NewRequest("GET", normalized, nil)
	applyDefaultHeaders(req)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, errors.New("fetch http status: " + resp.Status)
	}

	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	b, _ := io.ReadAll(io.LimitReader(resp.Body, 3_000_000))

	var text string
	if strings.Contains(ct, "text/plain") {
		text = normalizeWhitespace(html.UnescapeString(string(b)))
	} else {
		// default: treat as html
		page := string(b)
		text = normalizeWhitespace(stripHTML(page))
	}

	title := ""
	if strings.Contains(ct, "text/html") || ct == "" {
		title = extractTitle(string(b))
	}

	h := sha256.Sum256([]byte(text))
	hash := hex.EncodeToString(h[:])

	// Short snippet for display/storage (420 chars)
	snippet := text
	if len(snippet) > 420 {
		snippet = snippet[:420]
	}

	// Longer body for LLM context (first 3000 chars of clean text)
	body := text
	if len(body) > 3000 {
		body = body[:3000]
	}

	return &FetchResult{
		Title:     title,
		URL:       normalized,
		Text:      text,
		Snippet:   snippet,
		Body:      body,
		Hash:      hash,
		FetchedAt: time.Now(),
		Domain:    pu.Hostname(),
	}, nil
}

func applyDefaultHeaders(req *http.Request) {
	// Browser-like UA reduces 403 on many sites.
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
	req.Header.Set("Accept-Language", "de-DE,de;q=0.9,en;q=0.7")
	req.Header.Set("Connection", "close")
}

func normalizeResultURL(u string) string {
	u = strings.TrimSpace(u)
	if strings.HasPrefix(u, "//") {
		return "https:" + u
	}
	// DDG redirect: /l/?uddg=...
	if strings.HasPrefix(u, "/l/?") {
		return decodeDDGRedirect("https://duckduckgo.com" + u)
	}
	if strings.HasPrefix(u, "https://duckduckgo.com/l/?") || strings.HasPrefix(u, "http://duckduckgo.com/l/?") {
		return decodeDDGRedirect(u)
	}
	return u
}

func decodeDDGRedirect(ddg string) string {
	pu, err := url.Parse(ddg)
	if err != nil {
		return ddg
	}
	uddg := pu.Query().Get("uddg")
	if uddg == "" {
		return ddg
	}
	decoded, err := url.QueryUnescape(uddg)
	if err != nil {
		return uddg
	}
	return decoded
}

func extractTitle(page string) string {
	re := regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	m := re.FindStringSubmatch(page)
	if len(m) < 2 {
		return ""
	}
	return normalizeWhitespace(stripHTML(m[1]))
}

func stripHTML(s string) string {
	reSS := regexp.MustCompile(`(?is)<(script|style)[^>]*>.*?</(script|style)>`)
	s = reSS.ReplaceAllString(s, " ")
	reT := regexp.MustCompile(`(?is)<[^>]+>`)
	s = reT.ReplaceAllString(s, " ")
	return html.UnescapeString(s)
}

func normalizeWhitespace(s string) string {
	s = strings.ReplaceAll(s, "\u00a0", " ")
	re := regexp.MustCompile(`\s+`)
	s = re.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}
