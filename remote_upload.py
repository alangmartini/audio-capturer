"""
Upload finalized recordings to an exposer receiver.

The remote machine captures audio locally, then sends the WAV to a beelink
machine running exposer. Exposer can process the uploaded file with its
UPLOAD_HOOK.
"""

import os
import socket
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse


def _base_url(value):
    base = (value or "").strip().rstrip("/")
    if not base:
        raise ValueError("Remote upload URL is empty")
    return base + "/"


def _remote_path(filepath, remote_dir=None):
    host = socket.gethostname() or os.environ.get("COMPUTERNAME") or "remote"
    safe_host = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in host)
    path = Path(filepath)
    parts = []
    if remote_dir:
        parts.append(str(remote_dir).strip("/\\"))
    parts.extend([safe_host, path.parent.name, path.name])
    return "/".join(part for part in parts if part)


def upload_recording(filepath, server_url, remote_dir="audio-inbox", overwrite=True, timeout=120):
    """Stream a recording to exposer's /api/upload endpoint."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(path)

    target_path = _remote_path(path, remote_dir)
    query = urlencode({
        "path": target_path,
        "overwrite": "1" if overwrite else "0",
    })
    url = urlparse(urljoin(_base_url(server_url), "api/upload") + "?" + query)
    conn_cls = HTTPSConnection if url.scheme == "https" else HTTPConnection
    conn = conn_cls(url.hostname, url.port, timeout=timeout)

    request_target = url.path or "/"
    if url.query:
        request_target += "?" + url.query

    try:
        conn.putrequest("PUT", request_target)
        conn.putheader("content-type", "audio/wav")
        conn.putheader("content-length", str(path.stat().st_size))
        conn.endheaders()

        with path.open("rb") as body:
            while True:
                chunk = body.read(1024 * 1024)
                if not chunk:
                    break
                conn.send(chunk)

        response = conn.getresponse()
        response_body = response.read().decode("utf-8", errors="replace")
        if response.status >= 400:
            raise RuntimeError(f"Upload failed with HTTP {response.status}: {response_body}")
        return {
            "status": response.status,
            "remote_path": target_path,
            "body": response_body,
        }
    finally:
        conn.close()
