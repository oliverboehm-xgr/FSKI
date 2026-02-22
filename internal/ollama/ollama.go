package ollama

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
	Stream   bool      `json:"stream"`
}

type ChatResponse struct {
	Message Message `json:"message"`
	Done    bool    `json:"done"`
}

type Client struct {
	BaseURL string
	HTTP    *http.Client
}

func New(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTP: &http.Client{
			Timeout: 120 * time.Second,
		},
	}
}

func (c *Client) Chat(model string, messages []Message) (string, error) {
	model = strings.TrimSpace(model)
	if model == "" {
		model = "llama3.1:8b"
	}
	out, err := c.chatOnce(model, messages)
	if err == nil {
		return out, nil
	}
	if !isRecoverableModelError(err) {
		return "", err
	}
	alt := c.suggestFallbackModel(model)
	if alt == "" || strings.EqualFold(alt, model) {
		return "", err
	}
	out2, err2 := c.chatOnce(alt, messages)
	if err2 == nil {
		return out2, nil
	}
	return "", err
}

func (c *Client) chatOnce(model string, messages []Message) (string, error) {
	reqBody, _ := json.Marshal(ChatRequest{Model: model, Messages: messages, Stream: false})
	req, _ := http.NewRequest("POST", c.BaseURL+"/api/chat", bytes.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode >= 400 {
		msg := strings.TrimSpace(string(body))
		if msg == "" {
			msg = "ollama chat http status: " + resp.Status
		}
		return "", errors.New(msg)
	}
	var out ChatResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return "", err
	}
	return out.Message.Content, nil
}

func isRecoverableModelError(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(strings.TrimSpace(err.Error()))
	if s == "" {
		return false
	}
	keys := []string{"model", "not found", "unknown", "load", "manifest", "status 404", "status 500"}
	for _, k := range keys {
		if strings.Contains(s, k) {
			return true
		}
	}
	return false
}

func (c *Client) suggestFallbackModel(original string) string {
	models, err := c.ListModels()
	if err != nil || len(models) == 0 {
		return ""
	}
	prefer := []string{"llama3.1:8b", "qwen2.5:7b", "llama3.2:3b"}
	for _, m := range prefer {
		if _, ok := models[m]; ok && !strings.EqualFold(m, original) {
			return m
		}
	}
	for m := range models {
		if strings.EqualFold(strings.TrimSpace(m), strings.TrimSpace(original)) {
			continue
		}
		return strings.TrimSpace(m)
	}
	return ""
}

func (c *Client) Ping() error {
	req, _ := http.NewRequest("GET", c.BaseURL+"/api/tags", nil)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama status %d", resp.StatusCode)
	}

	return nil
}

func (c *Client) ListModels() (map[string]struct{}, error) {
	req, _ := http.NewRequest("GET", c.BaseURL+"/api/tags", nil)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var parsed struct {
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil {
		return nil, err
	}

	out := make(map[string]struct{}, len(parsed.Models))
	for _, m := range parsed.Models {
		name := strings.TrimSpace(m.Name)
		if name != "" {
			out[name] = struct{}{}
		}
	}
	return out, nil
}
