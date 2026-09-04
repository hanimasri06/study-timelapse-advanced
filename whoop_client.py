import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import base64
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class WhoopClient:
    AUTH_URL = "https://api.prod.whoop.com/oauth/oauth2/auth"
    TOKEN_URL = "https://api.prod.whoop.com/oauth/oauth2/token"
    API_BASE = "https://api.prod.whoop.com"
    DEFAULT_REDIRECT_URI = "https://localhost:8947/whoop/callback"
    SCOPES = "offline read:recovery read:sleep read:cycles"
    DEFAULT_HEADERS = {
        "Accept": "application/json, text/plain, */*",
        "User-Agent": "StudyTimelapseStudio/1.0 (Windows; WHOOP OAuth)",
    }

    def __init__(self, config_path=None, cache_ttl=600):
        self.config_path = config_path or os.path.join("sessions", "whoop_auth.json")
        self.cache_ttl = cache_ttl
        self.lock = threading.RLock()
        self.config = {
            "client_id": "",
            "client_secret": "",
            "redirect_uri": self.DEFAULT_REDIRECT_URI,
            "access_token": "",
            "refresh_token": "",
            "expires_at": 0,
            "scope": "",
            "cache": self.empty_metrics("Not connected"),
        }
        self.load()

    @staticmethod
    def empty_metrics(status="Not connected"):
        now = time.time()
        return {
            "available": False,
            "status": status,
            "last_synced_at": 0,
            "updated_at": "",
            "recovery": None,
            "sleep": None,
            "strain": None,
            "recovery_score": None,
            "sleep_performance": None,
            "strain_score": None,
            "source": "WHOOP",
            "generated_at": now,
        }

    def load(self):
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                with self.lock:
                    self.config.update(data)
        except (OSError, json.JSONDecodeError):
            pass

    def save(self):
        with self.lock:
            payload = dict(self.config)
            directory = os.path.dirname(self.config_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            temp_path = f"{self.config_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temp_path, self.config_path)

    def configure(self, client_id=None, client_secret=None, redirect_uri=None):
        changed = False
        with self.lock:
            if client_id is not None:
                value = client_id.strip()
                changed = changed or value != self.config.get("client_id", "")
                self.config["client_id"] = value
            if client_secret is not None:
                value = client_secret.strip()
                changed = changed or value != self.config.get("client_secret", "")
                self.config["client_secret"] = value
            if redirect_uri is not None:
                value = redirect_uri.strip() or self.DEFAULT_REDIRECT_URI
                changed = changed or value != self.config.get("redirect_uri", "")
                self.config["redirect_uri"] = value
        if changed:
            self.save()

    def is_configured(self):
        return bool(self.config.get("client_id") and self.config.get("client_secret") and self.config.get("redirect_uri"))

    def is_connected(self):
        return bool(self.config.get("access_token") or self.config.get("refresh_token"))

    def uses_local_http_callback(self):
        parsed = urllib.parse.urlparse(self.config.get("redirect_uri", ""))
        return parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost")

    def cached_metrics(self):
        cache = self.config.get("cache")
        if isinstance(cache, dict):
            return cache
        return self.empty_metrics("Not connected")

    def clear_tokens(self):
        self.config["access_token"] = ""
        self.config["refresh_token"] = ""
        self.config["expires_at"] = 0
        self.config["cache"] = self.empty_metrics("Not connected")
        self.save()

    def connect_via_browser(self, timeout=180):
        if not self.is_configured():
            raise RuntimeError("Add a WHOOP Client ID and Client Secret first.")

        redirect_uri = self.config["redirect_uri"]
        parsed = urllib.parse.urlparse(redirect_uri)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise RuntimeError("This redirect URI needs the manual code flow.")

        callback = {}
        state = self._new_oauth_state()
        path = parsed.path or "/"

        class CallbackHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                request = urllib.parse.urlparse(self.path)
                query = urllib.parse.parse_qs(request.query)
                callback["path"] = request.path
                callback["code"] = query.get("code", [""])[0]
                callback["state"] = query.get("state", [""])[0]
                callback["error"] = query.get("error", [""])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(
                    b"<html><body style='font-family:Segoe UI,Arial;background:#0d0f12;color:#f8fafc;'>"
                    b"<h2>WHOOP connected</h2><p>You can close this tab and return to Study Studio.</p>"
                    b"</body></html>"
                )

            def log_message(self, *_args):
                return

        server = HTTPServer(("127.0.0.1", parsed.port or 80), CallbackHandler)
        server.timeout = 1
        webbrowser.open(self.authorization_url(state))

        start = time.time()
        try:
            while time.time() - start < timeout and not callback:
                server.handle_request()
        finally:
            server.server_close()

        if not callback:
            raise RuntimeError("Timed out waiting for WHOOP OAuth callback.")
        if callback.get("path") != path:
            raise RuntimeError("Received an unexpected WHOOP callback path.")
        if callback.get("error"):
            raise RuntimeError(f"WHOOP authorization failed: {callback['error']}")
        if callback.get("state") != state:
            raise RuntimeError("WHOOP authorization state did not match.")
        if not callback.get("code"):
            raise RuntimeError("WHOOP did not return an authorization code.")

        token = self._token_post({
            "grant_type": "authorization_code",
            "code": callback["code"],
            "redirect_uri": redirect_uri,
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
        })
        self._apply_token_payload(token)
        self.config.pop("pending_oauth_state", None)
        self.save()
        return self.fetch_metrics(force=True)

    def begin_manual_authorization(self):
        if not self.is_configured():
            raise RuntimeError("Add a WHOOP Client ID and Client Secret first.")
        url = self.authorization_url()
        webbrowser.open(url)
        self.save()
        return url

    def finish_manual_authorization(self, code_or_url):
        code, state = self._extract_oauth_code(code_or_url)
        if not code:
            raise RuntimeError("Paste the code value or the full WHOOP redirect URL.")
        pending_state = self.config.get("pending_oauth_state", "")
        if pending_state and state and state != pending_state:
            raise RuntimeError("WHOOP authorization state did not match. Start Connect WHOOP again.")
        token = self._token_post({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config["redirect_uri"],
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
        })
        self._apply_token_payload(token)
        self.config.pop("pending_oauth_state", None)
        self.save()
        return self.fetch_metrics(force=True)

    def authorization_url(self, state=None):
        state = state or self._new_oauth_state()
        params = {
            "response_type": "code",
            "client_id": self.config["client_id"],
            "redirect_uri": self.config["redirect_uri"],
            "scope": self.SCOPES,
            "state": state,
        }
        return f"{self.AUTH_URL}?{urllib.parse.urlencode(params)}"

    def _new_oauth_state(self):
        state = secrets.token_urlsafe(24)
        self.config["pending_oauth_state"] = state
        return state

    def _extract_oauth_code(self, code_or_url):
        value = (code_or_url or "").strip()
        if not value:
            return "", ""
        if "code=" in value or value.startswith(("http://", "https://", "whoop://")):
            parsed = urllib.parse.urlparse(value)
            query_text = parsed.query if parsed.query else value.lstrip("?")
            query = urllib.parse.parse_qs(query_text)
            return query.get("code", [""])[0], query.get("state", [""])[0]
        if value.startswith("?"):
            query = urllib.parse.parse_qs(value[1:])
            return query.get("code", [""])[0], query.get("state", [""])[0]
        if "&" in value:
            code, rest = value.split("&", 1)
            query = urllib.parse.parse_qs(rest)
            return code, query.get("state", [""])[0]
        return value, ""

    def fetch_metrics(self, force=False):
        if not self.is_configured():
            metrics = self.empty_metrics("Not configured")
            self.config["cache"] = metrics
            return metrics
        if not self.is_connected():
            metrics = self.empty_metrics("Not connected")
            self.config["cache"] = metrics
            return metrics

        cached = self.cached_metrics()
        if not force and cached.get("last_synced_at") and time.time() - cached["last_synced_at"] < self.cache_ttl:
            return cached

        try:
            cycles = self._api_get("/developer/v2/cycle", {"limit": "1"})
            recoveries = self._api_get("/developer/v2/recovery", {"limit": "1"})
            sleeps = self._api_get("/developer/v2/activity/sleep", {"limit": "1"})
            metrics = self._build_metrics(cycles, recoveries, sleeps)
            self.config["cache"] = metrics
            self.save()
            return metrics
        except Exception as e:
            metrics = dict(cached) if isinstance(cached, dict) else self.empty_metrics()
            metrics["status"] = f"Sync failed: {e}"
            metrics["available"] = bool(metrics.get("recovery") or metrics.get("sleep") or metrics.get("strain"))
            self.config["cache"] = metrics
            self.save()
            return metrics

    def _build_metrics(self, cycles, recoveries, sleeps):
        recovery_record = self._latest_scored_record(recoveries)
        sleep_record = self._latest_scored_record(sleeps)
        cycle_record = self._latest_scored_record(cycles)

        recovery = self._parse_recovery(recovery_record)
        sleep = self._parse_sleep(sleep_record)
        strain = self._parse_strain(cycle_record)
        updated_values = [
            item.get("updated_at") for item in (recovery or {}, sleep or {}, strain or {}) if item.get("updated_at")
        ]

        available = bool(recovery or sleep or strain)
        return {
            "available": available,
            "status": "Synced" if available else "No scored WHOOP data yet",
            "last_synced_at": time.time(),
            "updated_at": max(updated_values) if updated_values else "",
            "recovery": recovery,
            "sleep": sleep,
            "strain": strain,
            "recovery_score": recovery.get("value") if recovery else None,
            "sleep_performance": sleep.get("value") if sleep else None,
            "strain_score": strain.get("value") if strain else None,
            "source": "WHOOP",
            "generated_at": time.time(),
        }

    def _latest_scored_record(self, payload):
        records = payload.get("records", []) if isinstance(payload, dict) else []
        if not isinstance(records, list):
            return None
        for record in records:
            if record.get("score_state") == "SCORED" and isinstance(record.get("score"), dict):
                return record
        return records[0] if records else None

    def _parse_recovery(self, record):
        if not record or record.get("score_state") != "SCORED":
            return None
        score = record.get("score", {})
        value = self._number(score.get("recovery_score"))
        if value is None:
            return None
        return {
            "label": "Recovery",
            "value": int(round(value)),
            "unit": "%",
            "max": 100,
            "zone": self._recovery_zone(value),
            "hrv": self._number(score.get("hrv_rmssd_milli")),
            "resting_hr": self._number(score.get("resting_heart_rate")),
            "updated_at": record.get("updated_at", ""),
        }

    def _parse_sleep(self, record):
        if not record or record.get("score_state") != "SCORED":
            return None
        score = record.get("score", {})
        value = self._number(score.get("sleep_performance_percentage"))
        if value is None:
            return None
        summary = score.get("stage_summary", {})
        needed = score.get("sleep_needed", {})
        asleep_ms = sum(self._number(summary.get(key)) or 0 for key in (
            "total_light_sleep_time_milli",
            "total_slow_wave_sleep_time_milli",
            "total_rem_sleep_time_milli",
        ))
        needed_ms = sum(self._number(needed.get(key)) or 0 for key in (
            "baseline_milli",
            "need_from_sleep_debt_milli",
            "need_from_recent_strain_milli",
            "need_from_recent_nap_milli",
        ))
        return {
            "label": "Sleep",
            "value": int(round(value)),
            "unit": "%",
            "max": 100,
            "duration_hours": round(asleep_ms / 3_600_000, 2) if asleep_ms else None,
            "needed_hours": round(max(0, needed_ms) / 3_600_000, 2) if needed_ms else None,
            "efficiency": self._number(score.get("sleep_efficiency_percentage")),
            "updated_at": record.get("updated_at", ""),
        }

    def _parse_strain(self, record):
        if not record or record.get("score_state") != "SCORED":
            return None
        score = record.get("score", {})
        value = self._number(score.get("strain"))
        if value is None:
            return None
        return {
            "label": "Strain",
            "value": round(value, 1),
            "unit": "",
            "max": 21,
            "average_hr": self._number(score.get("average_heart_rate")),
            "max_hr": self._number(score.get("max_heart_rate")),
            "updated_at": record.get("updated_at", ""),
        }

    def _api_get(self, path, params=None):
        token = self._access_token()
        url = f"{self.API_BASE}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = dict(self.DEFAULT_HEADERS)
        headers["Authorization"] = f"Bearer {token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            return self._read_json(request)
        except urllib.error.HTTPError as e:
            if e.code == 401 and self.config.get("refresh_token"):
                self._refresh_access_token()
                token = self.config["access_token"]
                headers["Authorization"] = f"Bearer {token}"
                request = urllib.request.Request(url, headers=headers)
                return self._read_json(request)
            raise

    def _access_token(self):
        if not self.config.get("access_token") or time.time() > float(self.config.get("expires_at", 0)) - 120:
            self._refresh_access_token()
        return self.config["access_token"]

    def _refresh_access_token(self):
        refresh_token = self.config.get("refresh_token")
        if not refresh_token:
            raise RuntimeError("WHOOP is not connected yet.")
        token = self._token_post({
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.config["client_id"],
            "client_secret": self.config["client_secret"],
            "scope": "offline",
        })
        self._apply_token_payload(token)
        self.save()

    def _token_post(self, payload):
        body = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(
            self.TOKEN_URL,
            data=body,
            headers={
                **self.DEFAULT_HEADERS,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        try:
            return self._read_json(request)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and payload.get("client_id") and payload.get("client_secret"):
                return self._token_post_basic(payload, e)
            raise self._http_error_with_body(e)

    def _token_post_basic(self, payload, original_error):
        basic_payload = dict(payload)
        client_id = basic_payload.pop("client_id", "")
        client_secret = basic_payload.pop("client_secret", "")
        body = urllib.parse.urlencode(basic_payload).encode("utf-8")
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
        request = urllib.request.Request(
            self.TOKEN_URL,
            data=body,
            headers={
                **self.DEFAULT_HEADERS,
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            method="POST",
        )
        try:
            return self._read_json(request)
        except urllib.error.HTTPError as e:
            raise self._http_error_with_body(e, original_error) from e

    def _read_json(self, request):
        with urllib.request.urlopen(request, timeout=20) as response:
            raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _http_error_with_body(self, error, original_error=None):
        body = ""
        try:
            body = error.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        prefix = f"HTTP {error.code} {error.reason}"
        if original_error is not None:
            prefix = f"HTTP {error.code} {error.reason} after Basic Auth fallback"
        if body:
            return RuntimeError(f"{prefix}: {body[:500]}")
        return RuntimeError(prefix)

    def _apply_token_payload(self, token):
        if not token.get("access_token"):
            raise RuntimeError("WHOOP token response did not include an access token.")
        self.config["access_token"] = token["access_token"]
        if token.get("refresh_token"):
            self.config["refresh_token"] = token["refresh_token"]
        self.config["expires_at"] = time.time() + int(token.get("expires_in", 3600))
        self.config["scope"] = token.get("scope", self.config.get("scope", ""))

    def _number(self, value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _recovery_zone(self, value):
        if value >= 67:
            return "green"
        if value >= 34:
            return "yellow"
        return "red"
