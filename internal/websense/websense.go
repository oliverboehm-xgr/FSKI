package websense

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
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
	Hash      string
	FetchedAt time.Time
	Domain    string
}

// DuckDuckGo HTML (quick & dirty, reicht für v0)
func Search(query string, k int) ([]SearchResult, error) {
	if k <= 0 {
		k = 5
	}
	q := url.QueryEscape(query)
	u := "https://duckduckgo.com/html/?q=" + q

	req, _ := http.NewRequest("GET", u, nil)
	req.Header.Set("User-Agent", "FSKI/0.1 (read-only)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, errors.New("search http status: " + resp.Status)
	}

	b, _ := io.ReadAll(io.LimitReader(resp.Body, 2_000_000))
	html := string(b)

	// Ergebnisse: <a rel="nofollow" class="result__a" href="...">TITLE</a>
	re := regexp.MustCompile(`<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>([^<]+)</a>`)
	m := re.FindAllStringSubmatch(html, k)

	out := make([]SearchResult, 0, len(m))
	for _, mm := range m {
		link := htmlUnescape(mm[1])
		title := htmlUnescape(mm[2])
		out = append(out, SearchResult{Title: title, URL: link})
	}
	return out, nil
}

func Fetch(rawURL string) (*FetchResult, error) {
	pu, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}

	req, _ := http.NewRequest("GET", rawURL, nil)
	req.Header.Set("User-Agent", "FSKI/0.1 (read-only)")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, errors.New("fetch http status: " + resp.Status)
	}

	b, _ := io.ReadAll(io.LimitReader(resp.Body, 3_000_000))
	html := string(b)

	title := extractTitle(html)
	text := normalizeWhitespace(stripHTML(html))

	h := sha256.Sum256([]byte(text))
	hash := hex.EncodeToString(h[:])

	snippet := text
	if len(snippet) > 400 {
		snippet = snippet[:400]
	}

	return &FetchResult{
		Title:     title,
		URL:       rawURL,
		Text:      text,
		Snippet:   snippet,
		Hash:      hash,
		FetchedAt: time.Now(),
		Domain:    pu.Hostname(),
	}, nil
}

func extractTitle(html string) string {
	re := regexp.MustCompile(`(?is)<title[^>]*>(.*?)</title>`)
	m := re.FindStringSubmatch(html)
	if len(m) < 2 {
		return ""
	}
	return normalizeWhitespace(stripHTML(m[1]))
}

func stripHTML(s string) string {
	// scripts/styles raus
	reSS := regexp.MustCompile(`(?is)<(script|style)[^>]*>.*?</\1>`)
	s = reSS.ReplaceAllString(s, " ")

	// tags raus
	reT := regexp.MustCompile(`(?is)<[^>]+>`)
	s = reT.ReplaceAllString(s, " ")

	return htmlUnescape(s)
}

func normalizeWhitespace(s string) string {
	s = strings.ReplaceAll(s, "\u00a0", " ")
	re := regexp.MustCompile(`\s+`)
	s = re.ReplaceAllString(s, " ")
	return strings.TrimSpace(s)
}

func htmlUnescape(s string) string {
	r := strings.NewReplacer(
		"&amp;", "&",
		"&lt;", "<",
		"&gt;", ">",
		"&quot;", `"`,
		"&#39;", "'",
	)
	return r.Replace(s)
}