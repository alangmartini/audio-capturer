"""
HTTP transport to an exposer receiver.

The client records audio locally and sends it to a beelink running exposer,
which processes it with its UPLOAD_HOOK.  Reaching that machine from a
corporate network normally means going through exposer's Cloudflare Worker
(`https://exposer.<account>.workers.dev`), which sits in front of the tunnel
and demands HTTP Basic auth — so every request here carries credentials when
they are configured.  Direct LAN use (`http://beelink:8080`) needs none, and
passing no password simply omits the header.

Only the standard library is used: this module has to run on a locked-down
work machine where installing packages may not be an option.
"""

import base64
import json
import os
import socket
from http.client import HTTPConnection, HTTPSConnection
from pathlib import Path
from urllib.parse import urlencode, urljoin, urlparse

from remote_common import client_hostname, log_event

DEFAULT_TIMEOUT = 120
UPLOAD_CHUNK = 1024 * 1024

# Exposer's Worker defaults to this Basic-auth username (PROXY_USER).
DEFAULT_PROXY_USER = "exposer"


def _base_url(value):
    base = (value or "").strip().rstrip("/")
    if not base:
        raise ValueError("Remote upload URL is empty")
    return base + "/"


def resolve_password(explicit=None):
    """
    Find the proxy password: explicit argument first, then the environment.

    Kept in one place so no caller has to remember the env var names, and so
    the value never needs to be written into the config file if the user
    prefers to keep it in the environment.
    """
    for candidate in (
        explicit,
        os.environ.get("REMOTE_UPLOAD_PASSWORD"),
        os.environ.get("PROXY_PASSWORD"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return None


class RemoteEndpoint:
    """One exposer receiver, with the credentials needed to talk to it."""

    def __init__(self, server_url, user=None, password=None, timeout=DEFAULT_TIMEOUT):
        self.base = _base_url(server_url)
        self.user = (user or DEFAULT_PROXY_USER).strip()
        self.password = resolve_password(password)
        self.timeout = timeout

    # ── internals ────────────────────────────────────────────────────────
    def _target(self, api_path, params=None):
        url = urlparse(urljoin(self.base, api_path))
        target = url.path or "/"
        query = url.query
        if params:
            extra = urlencode(params)
            query = f"{query}&{extra}" if query else extra
        if query:
            target += "?" + query
        return url, target

    def _connect(self, url):
        cls = HTTPSConnection if url.scheme == "https" else HTTPConnection
        port = url.port or (443 if url.scheme == "https" else 80)
        return cls(url.hostname, port, timeout=self.timeout)

    def _auth_headers(self):
        # No password configured means a direct LAN endpoint with no auth in
        # front of it; sending an empty Basic header would only confuse it.
        if not self.password:
            return {}
        raw = f"{self.user}:{self.password}".encode("utf-8")
        return {"authorization": "Basic " + base64.b64encode(raw).decode("ascii")}

    def _describe_failure(self, status, body):
        """Turn a bare status code into something a user can act on."""
        if status in (401, 407):
            return (
                f"HTTP {status}: the remote rejected the credentials. Set the "
                f"remote upload password (Settings -> Remote Processing, or the "
                f"REMOTE_UPLOAD_PASSWORD environment variable)."
            )
        if status == 404:
            return f"HTTP 404: not found on the remote."
        return f"HTTP {status}: {body[:400]}"

    # ── operations ───────────────────────────────────────────────────────
    def request(self, method, api_path, params=None, expect=(200,)):
        """Perform a small request and return (status, body-bytes)."""
        url, target = self._target(api_path, params)
        conn = self._connect(url)
        try:
            conn.request(method, target, headers=self._auth_headers())
            response = conn.getresponse()
            body = response.read()
            if expect and response.status not in expect and response.status >= 400:
                raise RuntimeError(
                    self._describe_failure(
                        response.status, body.decode("utf-8", errors="replace")
                    )
                )
            return response.status, body
        finally:
            conn.close()

    def download(self, remote_path, missing_ok=False):
        """
        Fetch a file from the share.  Returns bytes, or None when the file is
        not there yet and `missing_ok` — the normal case while polling for a
        status document the host has not written yet.
        """
        status, body = self.request(
            "GET", "api/download", {"path": remote_path},
            expect=(200, 404) if missing_ok else (200,),
        )
        if status == 404:
            return None
        return body

    def download_json(self, remote_path, missing_ok=True):
        """Fetch and parse a JSON file, tolerating a torn read."""
        raw = self.download(remote_path, missing_ok=missing_ok)
        if raw is None:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            # The host rewrites status documents atomically, so this should
            # not happen — but a truncated proxy response shouldn't kill a
            # 40-minute job either.  The caller retries on the next poll.
            log_event("remote_json_parse_failed", level="warn", path=remote_path,
                      bytes=len(raw))
            return None

    def test_connection(self):
        """Check that this process can reach exposer and is authenticated."""
        url, _ = self._target("api/list", {"path": ""})
        port = url.port or (443 if url.scheme == "https" else 80)
        addresses = sorted(
            {item[4][0] for item in socket.getaddrinfo(url.hostname, port, type=socket.SOCK_STREAM)}
        )
        socket.create_connection((url.hostname, port), timeout=self.timeout).close()
        status, _ = self.request("GET", "api/list", {"path": ""})
        return {
            "host": url.hostname,
            "port": port,
            "addresses": addresses,
            "status": status,
            "authenticated": bool(self.password),
        }

    def upload_file(self, local_path, remote_path, content_type="application/octet-stream",
                    overwrite=True, progress_callback=None):
        """
        Stream a local file to `remote_path` on the share.

        `progress_callback(sent_bytes, total_bytes)` is called as the body
        goes out — on a VPN this upload is often the slowest part of the whole
        round trip, so the user needs to see it move.
        """
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(path)
        total = path.stat().st_size

        url, target = self._target(
            "api/upload", {"path": remote_path, "overwrite": "1" if overwrite else "0"}
        )
        conn = self._connect(url)
        try:
            conn.putrequest("PUT", target)
            conn.putheader("content-type", content_type)
            conn.putheader("content-length", str(total))
            for key, value in self._auth_headers().items():
                conn.putheader(key, value)
            conn.endheaders()

            sent = 0
            with path.open("rb") as body:
                while True:
                    chunk = body.read(UPLOAD_CHUNK)
                    if not chunk:
                        break
                    conn.send(chunk)
                    sent += len(chunk)
                    if progress_callback:
                        progress_callback(sent, total)

            response = conn.getresponse()
            response_body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(self._describe_failure(response.status, response_body))
            return {
                "status": response.status,
                "remote_path": remote_path,
                "bytes": total,
                "body": response_body,
            }
        finally:
            conn.close()

    def upload_bytes(self, data, remote_path, content_type="application/json"):
        """Upload a small in-memory payload (used for job manifests)."""
        url, target = self._target("api/upload", {"path": remote_path, "overwrite": "1"})
        conn = self._connect(url)
        try:
            conn.putrequest("PUT", target)
            conn.putheader("content-type", content_type)
            conn.putheader("content-length", str(len(data)))
            for key, value in self._auth_headers().items():
                conn.putheader(key, value)
            conn.endheaders()
            conn.send(data)
            response = conn.getresponse()
            body = response.read().decode("utf-8", errors="replace")
            if response.status >= 400:
                raise RuntimeError(self._describe_failure(response.status, body))
            return {"status": response.status, "remote_path": remote_path}
        finally:
            conn.close()


# ─── Backwards-compatible module functions ────────────────────────────────
#
# capture.py and app.py already call these; they keep working unchanged, and
# gain auth by passing the new keyword arguments.

def _remote_path(filepath, remote_dir=None):
    path = Path(filepath)
    parts = []
    if remote_dir:
        parts.append(str(remote_dir).strip("/\\"))
    parts.extend([client_hostname(), path.parent.name, path.name])
    return "/".join(part for part in parts if part)


def test_connection(server_url, timeout=10, user=None, password=None):
    """Check whether this Python process can reach exposer."""
    return RemoteEndpoint(server_url, user=user, password=password, timeout=timeout).test_connection()


def upload_recording(filepath, server_url, remote_dir="audio-inbox", overwrite=True,
                     timeout=DEFAULT_TIMEOUT, user=None, password=None,
                     progress_callback=None):
    """Stream a recording to exposer's /api/upload endpoint."""
    endpoint = RemoteEndpoint(server_url, user=user, password=password, timeout=timeout)
    target_path = _remote_path(filepath, remote_dir)
    result = endpoint.upload_file(
        filepath, target_path, content_type="audio/wav",
        overwrite=overwrite, progress_callback=progress_callback,
    )
    log_event("remote_upload_complete", remote_path=target_path, bytes=result["bytes"])
    return result
