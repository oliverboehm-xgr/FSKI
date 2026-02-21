package ui

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"sync"
	"time"
)

type Message struct {
	ID        int64  `json:"id"`
	CreatedAt string `json:"created_at"`
	Kind      string `json:"kind"` // auto|reply|think|user
	Text      string `json:"text"`
	Rating    *int   `json:"rating,omitempty"` // -1,0,1
}

type Server struct {
	addr string

	// callbacks into your kernel/app
	ListMessages func(limit int) ([]Message, error)
	SendText     func(text string) (Message, error)
	RateMessage  func(messageID int64, value int) error
	Caught       func(messageID int64) error
	Status       func() (any, error)

	b *broker
}

func New(addr string) *Server {
	return &Server{
		addr: addr,
		b:    newBroker(),
	}
}

// PublishMessage pushes a message to all SSE subscribers.
func (s *Server) PublishMessage(m Message) {
	if s == nil || s.b == nil {
		return
	}
	s.b.publish("message", m)
}

// PublishStatus pushes a status snapshot to SSE subscribers.
func (s *Server) PublishStatus(st any) {
	if s == nil || s.b == nil {
		return
	}
	s.b.publish("status", st)
}

func (s *Server) Run(ctx context.Context) error {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		fmt.Fprint(w, indexHTML)
	})

	mux.HandleFunc("/api/messages", func(w http.ResponseWriter, r *http.Request) {
		limit := 50
		if q := r.URL.Query().Get("limit"); q != "" {
			if v, err := strconv.Atoi(q); err == nil && v >= 1 && v <= 500 {
				limit = v
			}
		}
		if s.ListMessages == nil {
			http.Error(w, "ListMessages not configured", http.StatusInternalServerError)
			return
		}
		msgs, err := s.ListMessages(limit)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, msgs)
	})

	mux.HandleFunc("/api/status", func(w http.ResponseWriter, r *http.Request) {
		if s.Status == nil {
			http.Error(w, "Status not configured", http.StatusInternalServerError)
			return
		}
		st, err := s.Status()
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		writeJSON(w, st)
	})

	mux.HandleFunc("/api/send", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if s.SendText == nil {
			http.Error(w, "SendText not configured", http.StatusInternalServerError)
			return
		}
		var body struct {
			Text string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		body.Text = trim(body.Text)
		if body.Text == "" {
			http.Error(w, "empty", http.StatusBadRequest)
			return
		}
		msg, err := s.SendText(body.Text)
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		// Push via SSE too
		s.PublishMessage(msg)
		writeJSON(w, msg)
	})

	mux.HandleFunc("/api/rate", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if s.RateMessage == nil {
			http.Error(w, "RateMessage not configured", http.StatusInternalServerError)
			return
		}
		var body struct {
			MessageID int64 `json:"message_id"`
			Value     int   `json:"value"` // 1,0,-1
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if body.MessageID <= 0 || (body.Value != 1 && body.Value != 0 && body.Value != -1) {
			http.Error(w, "bad payload", http.StatusBadRequest)
			return
		}
		if err := s.RateMessage(body.MessageID, body.Value); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("/api/caught", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		if s.Caught == nil {
			http.Error(w, "Caught not configured", http.StatusInternalServerError)
			return
		}
		var body struct {
			MessageID int64 `json:"message_id"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			http.Error(w, "bad json", http.StatusBadRequest)
			return
		}
		if body.MessageID <= 0 {
			http.Error(w, "bad payload", http.StatusBadRequest)
			return
		}
		if err := s.Caught(body.MessageID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	// SSE stream
	mux.HandleFunc("/api/stream", func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "stream unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		ch, cancel := s.b.subscribe()
		defer cancel()

		// initial keepalive
		fmt.Fprint(w, "event: ping\ndata: {}\n\n")
		flusher.Flush()

		keep := time.NewTicker(15 * time.Second)
		defer keep.Stop()

		for {
			select {
			case <-r.Context().Done():
				return
			case <-ctx.Done():
				return
			case msg := <-ch:
				_, _ = w.Write(msg)
				flusher.Flush()
			case <-keep.C:
				fmt.Fprint(w, "event: ping\ndata: {}\n\n")
				flusher.Flush()
			}
		}
	})

	srv := &http.Server{
		Addr:    s.addr,
		Handler: mux,
	}

	go func() {
		<-ctx.Done()
		_ = srv.Shutdown(context.Background())
	}()

	return srv.ListenAndServe()
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(v)
}

func trim(s string) string {
	// minimal, avoids extra deps
	for len(s) > 0 && (s[0] == ' ' || s[0] == '\n' || s[0] == '\t' || s[0] == '\r') {
		s = s[1:]
	}
	for len(s) > 0 && (s[len(s)-1] == ' ' || s[len(s)-1] == '\n' || s[len(s)-1] == '\t' || s[len(s)-1] == '\r') {
		s = s[:len(s)-1]
	}
	return s
}

type broker struct {
	mu   sync.Mutex
	subs map[chan []byte]struct{}
}

func newBroker() *broker {
	return &broker{subs: map[chan []byte]struct{}{}}
}

func (b *broker) subscribe() (chan []byte, func()) {
	ch := make(chan []byte, 16)
	b.mu.Lock()
	b.subs[ch] = struct{}{}
	b.mu.Unlock()
	return ch, func() {
		b.mu.Lock()
		delete(b.subs, ch)
		b.mu.Unlock()
		close(ch)
	}
}

func (b *broker) publish(event string, payload any) {
	b.mu.Lock()
	defer b.mu.Unlock()
	bb, _ := json.Marshal(payload)
	msg := []byte(fmt.Sprintf("event: %s\ndata: %s\n\n", event, string(bb)))
	for ch := range b.subs {
		select {
		case ch <- msg:
		default:
			// drop if slow consumer
		}
	}
}

const indexHTML = `<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Bunny UI</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #0b0b0c; color: #eaeaea; }
    .wrap { display: grid; grid-template-columns: 1fr 320px; height: 100vh; }
    .chat { padding: 16px; overflow: auto; }
    .side { border-left: 1px solid #222; padding: 16px; overflow: auto; background: #0f0f11; }
    .msg { background: #131316; border: 1px solid #242428; border-radius: 12px; padding: 12px; margin: 10px 0; }
    .msg.user { background:#0f1a12; border-color:#21402b; margin-left: 64px; }
    .msg.reply,.msg.auto,.msg.think { margin-right: 64px; }
    .meta { opacity: 0.7; font-size: 12px; display:flex; justify-content: space-between; gap: 12px; }
    .text { white-space: pre-wrap; line-height: 1.35; margin-top: 8px; }
    .btns { margin-top: 10px; display:flex; gap: 8px; align-items:center; }
    button { background:#1b1b20; border:1px solid #2b2b33; color:#eaeaea; border-radius:10px; padding:6px 10px; cursor:pointer; }
    button:hover { background:#24242a; }
    button:disabled { opacity:0.45; cursor: default; }
    .ack { font-size: 12px; opacity: 0.75; margin-left: 6px; }
    .row { display:flex; gap: 10px; padding: 12px; border-top: 1px solid #222; }
    input { flex:1; background:#101012; border:1px solid #2b2b33; border-radius:10px; padding:10px; color:#eaeaea; }
    .tag { font-size: 11px; padding:2px 8px; border:1px solid #2b2b33; border-radius:999px; }
    pre { white-space: pre-wrap; font-size: 12px; opacity: 0.9; }
  </style>
</head>
<body>
  <div class="wrap">
    <div>
      <div id="chat" class="chat"></div>
      <div class="row">
        <input id="inp" placeholder="Schreib an Bunny…" />
        <button id="send">Senden</button>
      </div>
    </div>
    <div class="side">
      <div style="display:flex; justify-content: space-between; align-items:center;">
        <div><b>Status</b></div>
        <button id="refresh">↻</button>
      </div>
      <pre id="status">(lädt…)</pre>
      <div style="margin-top:16px; opacity:0.8; font-size:12px;">
        Feedback: 👍 = gut, 😐 = ok, 👎 = schlecht, ❌ = gelogen / grob falsch
      </div>
    </div>
  </div>
<script>
  const chat = document.getElementById('chat');
  const inp = document.getElementById('inp');
  const sendBtn = document.getElementById('send');
  const statusEl = document.getElementById('status');
  const refreshBtn = document.getElementById('refresh');

  function esc(s){ return (s||'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;'); }

  function scrollBottom(){
    requestAnimationFrame(()=>{ chat.scrollTop = chat.scrollHeight; });
  }

  function addMsg(m){
    chat.appendChild(renderMsg(m));
    scrollBottom();
  }

  async function loadMessages(){
    const res = await fetch('/api/messages?limit=80');
    const msgs = await res.json();
    chat.innerHTML = '';
    msgs.reverse().forEach(addMsg);
    scrollBottom();
  }

  function renderMsg(m){
    const div = document.createElement('div');
    div.className = 'msg ' + (m.kind||'');
    div.dataset.id = m.id;
    const rated = (m.rating === 1 || m.rating === 0 || m.rating === -1);
    div.innerHTML =
      '<div class="meta">'+
        '<div>'+
          '<span class="tag">'+esc(m.kind||'')+'</span>'+
          '<span style="margin-left:8px;">#'+m.id+'</span>'+
        '</div>'+
        '<div>'+esc(m.created_at||'')+'</div>'+
      '</div>'+
      '<div class="text">'+esc(m.text||'')+'</div>'+
      ((m.kind==='user') ? '' :
      '<div class="btns">'+
        '<button data-v="1" '+(rated ? 'disabled' : '')+'>👍</button>'+
        '<button data-v="0" '+(rated ? 'disabled' : '')+'>😐</button>'+
        '<button data-v="-1" '+(rated ? 'disabled' : '')+'>👎</button>'+
        '<button data-c="1">❌ caught</button>'+
        '<span class="ack">'+(rated ? '✓ gespeichert' : '')+'</span>'+
      '</div>');

    div.querySelectorAll('button[data-v]').forEach(b=>{
      b.addEventListener('click', async ()=>{
        const v = parseInt(b.dataset.v,10);
        const ack = div.querySelector('.ack');
        b.disabled = true;
        div.querySelectorAll('button[data-v]').forEach(x=>x.disabled=true);
        const res = await fetch('/api/rate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message_id:m.id, value:v})});
        if(res.ok){
          if(ack) ack.textContent = '✓ gespeichert';
        } else {
          if(ack) ack.textContent = '✗ Fehler';
          div.querySelectorAll('button[data-v]').forEach(x=>x.disabled=false);
        }
      });
    });
    const caughtBtn = div.querySelector('button[data-c]');
    if(caughtBtn){
      caughtBtn.addEventListener('click', async ()=>{
        const ack = div.querySelector('.ack');
        caughtBtn.disabled = true;
        const res = await fetch('/api/caught', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message_id:m.id})});
        if(res.ok){
          if(ack) ack.textContent = '✓ caught';
        } else {
          if(ack) ack.textContent = '✗ Fehler';
          caughtBtn.disabled = false;
        }
      });
    }
    return div;
  }

  async function loadStatus(){
    const res = await fetch('/api/status');
    const st = await res.json();
    statusEl.textContent = JSON.stringify(st, null, 2);
  }

  async function send(){
    const t = (inp.value||'').trim();
    if(!t) return;
    inp.value='';
    await fetch('/api/send', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({text:t})});
  }

  sendBtn.addEventListener('click', send);
  inp.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ send(); }});
  refreshBtn.addEventListener('click', ()=>{ loadStatus(); });

  (async ()=>{
    await loadMessages();
    await loadStatus();

    const es = new EventSource('/api/stream');
    es.addEventListener('message', (ev)=>{
      const m = JSON.parse(ev.data);
      addMsg(m);
    });
    es.addEventListener('status', (ev)=>{
      const st = JSON.parse(ev.data);
      statusEl.textContent = JSON.stringify(st, null, 2);
    });
  })();
</script>
</body>
</html>`
