from __future__ import annotations

import json
import ssl
import os
import time
import sqlite3
import inspect
import datetime
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple



def _log_llm_call(organ: str, model: str, purpose: str, started_at: str, duration_ms: float, ok: int, error: str) -> None:
    db_path = os.environ.get("BUNNY_DB_PATH", "")
    if not db_path:
        return
    try:
        con = sqlite3.connect(db_path)
        try:
            con.execute(
                "INSERT INTO llm_calls(organ,model,purpose,started_at,duration_ms,ok,error) VALUES(?,?,?,?,?,?,?)",
                (organ or "", model or "", purpose or "", started_at, float(duration_ms), int(ok), error or ""),
            )
            con.commit()
        finally:
            con.close()
    except Exception:
        return


@dataclass
class HttpResponse:
    status: int
    url: str
    text: str
    content_type: str = ""


def http_get(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 20.0) -> HttpResponse:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
            ct = resp.headers.get("Content-Type", "") or ""
            data = resp.read()
            # Best-effort decode
            try:
                txt = data.decode("utf-8", errors="replace")
            except Exception:
                txt = data.decode(errors="replace")
            return HttpResponse(status=int(resp.status), url=resp.geturl(), text=txt, content_type=ct)
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            txt = data.decode("utf-8", errors="replace")
        except Exception:
            txt = data.decode(errors="replace")
        return HttpResponse(status=int(e.code), url=url, text=txt, content_type=e.headers.get("Content-Type","") or "")
    except Exception as e:
        # Use status 0 to indicate transport error
        return HttpResponse(status=0, url=url, text=f"{type(e).__name__}: {e}", content_type="")


def http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: float = 60.0) -> Tuple[int, str]:
    started = time.time()
    started_at = datetime.datetime.utcnow().isoformat(timespec='seconds')+'Z'
    organ = ''
    try:
        for fr in inspect.stack()[1:8]:
            mod = fr.frame.f_globals.get('__name__','')
            if mod.startswith('app.organs.'):
                organ = mod.split('.')[-1]
                break
        if not organ:
            organ = inspect.stack()[1].frame.f_globals.get('__name__','')
    except Exception:
        organ = ''

    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
            data = resp.read()
            try:
                txt = data.decode("utf-8", errors="replace")
            except Exception:
                txt = data.decode(errors="replace")
            dur_ms = (time.time()-started)*1000.0
            _log_llm_call(organ, str(payload.get('model','')), str(payload.get('purpose','')), started_at, dur_ms, 1, '')
            return int(resp.status), txt
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            txt = data.decode("utf-8", errors="replace")
        except Exception:
            txt = data.decode(errors="replace")
        dur_ms = (time.time()-started)*1000.0
        _log_llm_call(organ, str(payload.get('model','')), str(payload.get('purpose','')), started_at, dur_ms, 0, f'HTTPError {e.code}')
        return int(e.code), txt
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"
