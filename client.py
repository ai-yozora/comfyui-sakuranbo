"""A small client for the Sakuranbo AI API, using only the Python standard library."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable


def _version() -> str:
    """The version in pyproject.toml, which ships with the node."""
    try:
        text = (Path(__file__).resolve().parent / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


VERSION = _version()
USER_AGENT = f"sakuranbo-comfyui/{VERSION}"
DEFAULT_BASE_URL = "https://api.sakuranbo.ai"
API_KEY_ENV = "SAKURANBO_API_KEY"
BASE_URL_ENV = "SAKURANBO_BASE_URL"

FINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "expired"})
RETRY_STATUSES = frozenset({429, 502, 503})
RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 60
MODELS_TTL_SECONDS = 600
MODELS_TIMEOUT_SECONDS = 5


class SakuranboError(RuntimeError):
    """An error to show to the ComfyUI user."""


class JobTimeout(SakuranboError):
    """The job was still running when the node stopped waiting."""


def base_url() -> str:
    return (os.environ.get(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def api_key(value: str) -> str:
    """The key typed into the node, or else the SAKURANBO_API_KEY environment variable."""
    key = (value or "").strip() or os.environ.get(API_KEY_ENV, "").strip()
    if not key:
        raise SakuranboError(
            f"No API key. Set the {API_KEY_ENV} environment variable before starting ComfyUI, "
            "or enter a key in the node's api_key field."
        )
    return key


def idempotency_key() -> str:
    return f"comfyui-{uuid.uuid4()}"


def _retry_wait(headers: Any, attempt: int) -> float:
    try:
        seconds = float(headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        seconds = 2.0**attempt
    return max(0.0, min(seconds, MAX_RETRY_WAIT_SECONDS))


def _api_error(status: int, payload: bytes) -> SakuranboError:
    try:
        error = json.loads(payload)["error"]
        return error_from(error, f"HTTP {status}")
    except (ValueError, KeyError, TypeError):
        return SakuranboError(f"Sakuranbo AI returned HTTP {status}.")


def error_from(error: dict, fallback: str) -> SakuranboError:
    """Turn an API error object (a response's or a job's `error`) into a message."""
    code = error.get("code") or fallback
    message = error.get("message") or "The request failed."
    request_id = error.get("request_id")
    suffix = f" (request {request_id})" if request_id else ""
    return SakuranboError(f"Sakuranbo AI: {code}: {message}{suffix}")


def request(
    method: str,
    path: str,
    *,
    key: str | None = None,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = 30,
) -> tuple[int, Any]:
    """Send a JSON request to the API and return (status, decoded body).

    Retries 429, 502, 503, and network errors, waiting for Retry-After. POST requests
    must carry an Idempotency-Key so that a retry never charges twice.
    """
    url = path if path.startswith("https://") or path.startswith("http://") else base_url() + path
    data = None if body is None else json.dumps(body).encode()
    sent = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        sent["Content-Type"] = "application/json"
    if key:
        sent["Authorization"] = f"Bearer {key}"
    sent.update(headers or {})
    for attempt in range(RETRIES + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=sent)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = response.read()
                return response.status, json.loads(payload) if payload else None
        except urllib.error.HTTPError as error:
            payload = error.read()
            if error.code in RETRY_STATUSES and attempt < RETRIES:
                time.sleep(_retry_wait(error.headers, attempt))
                continue
            raise _api_error(error.code, payload) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt < RETRIES:
                time.sleep(_retry_wait({}, attempt))
                continue
            host = urllib.parse.urlsplit(url).netloc
            raise SakuranboError(f"Could not reach {host}: {error}") from None
    raise SakuranboError("Sakuranbo AI did not respond.")


def download(url: str, timeout: float = 300) -> bytes:
    """Download an output file. The signed URL needs no API key; never log it."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in RETRY_STATUSES and attempt < RETRIES:
                time.sleep(_retry_wait(error.headers, attempt))
                continue
            raise SakuranboError(f"Could not download the output (HTTP {error.code}).") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt < RETRIES:
                time.sleep(_retry_wait({}, attempt))
                continue
            raise SakuranboError(f"Could not download the output: {error}") from None
    raise SakuranboError("Could not download the output.")


def wait_for_job(
    path: str,
    key: str,
    *,
    interval: float,
    timeout: float,
    on_progress: Callable[[int | None], None] = lambda _: None,
    check: Callable[[], None] = lambda: None,
) -> dict:
    """Poll a job (GET path) until it reaches a final status, and return the completed job.

    Raises for a job that did not complete, and JobTimeout after timeout seconds.
    check() runs about twice a second while waiting and may raise to stop waiting.
    """
    deadline = time.monotonic() + timeout
    while True:
        check()
        _, job = request("GET", path, key=key)
        on_progress(job.get("progress"))
        status = job.get("status")
        if status in FINAL_STATUSES:
            if status == "completed":
                return job
            if job.get("error"):
                raise error_from(job["error"], status)
            raise SakuranboError(f"Sakuranbo AI: the job {job.get('id')} ended as {status}.")
        if time.monotonic() > deadline:
            raise JobTimeout(f"Sakuranbo AI: the job {job.get('id')} did not finish in {int(timeout)} seconds.")
        resume = time.monotonic() + interval
        while (left := resume - time.monotonic()) > 0:
            check()
            time.sleep(min(0.5, left))


_models_cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}


def list_models(modality: str, task: str) -> list[dict]:
    """The models for a modality and a task, cached for a few minutes. [] when unreachable."""
    cache_key = (modality, task)
    cached = _models_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < MODELS_TTL_SECONDS:
        return cached[1]
    query = urllib.parse.urlencode({"modality": modality, "task": task})
    req = urllib.request.Request(
        f"{base_url()}/v1/models?{query}", headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=MODELS_TIMEOUT_SECONDS) as response:
            models = [m for m in json.loads(response.read()).get("data", []) if m.get("id")]
    except (urllib.error.URLError, TimeoutError, ConnectionError, ValueError):
        return cached[1] if cached else []
    _models_cache[cache_key] = (time.monotonic(), models)
    return models
