"""Execute code in the live molab marimo session (Windows-friendly clone of execute-code.sh).

The molab proxy sits behind Cloudflare (error 1010) and rejects non-browser
clients, so every request carries browser headers. Auth via Bearer token.

Usage:
    set MOLAB_URL=https://sb-....sb.molab.run/
    set MOLAB_TOKEN=<token>
    python scripts/remote/molab_exec.py -c "print('hi')"
    python scripts/remote/molab_exec.py script.py
    python scripts/remote/molab_exec.py -            # read code from stdin

Do NOT commit real tokens — pass via env vars.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def headers(token: str, session_id: str = "") -> dict:
    h = {"User-Agent": UA, "Accept": "application/json", "Authorization": f"Bearer {token}"}
    if session_id:
        h["Marimo-Session-Id"] = session_id
    return h


def get_session(base: str, token: str) -> tuple[str, dict]:
    req = urllib.request.Request(base + "/api/sessions", headers=headers(token))
    sessions = json.load(urllib.request.urlopen(req, timeout=20))
    ids = list(sessions)
    if not ids:
        raise RuntimeError("no active sessions — open the notebook in the browser")
    if len(ids) > 1:
        raise RuntimeError(f"multiple sessions {ids} — extend this script with --session")
    return ids[0], sessions[ids[0]]


def execute(base: str, token: str, session_id: str, code: str) -> int:
    body = json.dumps({"code": code}).encode()
    req = urllib.request.Request(
        base + "/api/kernel/execute",
        data=body,
        headers={**headers(token, session_id), "Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=300)
    event, exit_code = "", 0
    for raw in resp:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            payload = json.loads(line[len("data:"):].strip())
            if event == "stdout":
                sys.stdout.write(payload.get("data", ""))
            elif event == "stderr":
                sys.stderr.write(payload.get("data", ""))
            elif event == "done":
                if not payload.get("success", True):
                    exit_code = 1
                else:
                    out = (payload.get("output") or {}).get("data")
                    if out:
                        print(out)
                return exit_code
    raise RuntimeError("stream ended without a done event")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    base = os.environ.get("MOLAB_URL", "").rstrip("/")
    token = os.environ.get("MOLAB_TOKEN", "")
    if not base or not token:
        print("set MOLAB_URL and MOLAB_TOKEN env vars", file=sys.stderr)
        sys.exit(2)
    args = sys.argv[1:]
    if args and args[0] == "-c":
        code = args[1]
    elif args and args[0] != "-":
        code = open(args[0]).read()
    else:
        code = sys.stdin.read()
    session_id, info = get_session(base, token)
    print(f"[session {session_id} {info.get('path', '')}]", file=sys.stderr)
    sys.exit(execute(base, token, session_id, code))


if __name__ == "__main__":
    main()
