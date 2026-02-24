from __future__ import annotations

import json
import ssl
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


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
            return int(resp.status), txt
    except urllib.error.HTTPError as e:
        data = e.read()
        try:
            txt = data.decode("utf-8", errors="replace")
        except Exception:
            txt = data.decode(errors="replace")
        return int(e.code), txt
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"
