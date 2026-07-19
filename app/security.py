"""
security.py – API-key authentication for AITrader's mutating endpoints.

The key is read from the ``AITRADER_API_KEY`` environment variable at request
time.  Behaviour is fail-closed: if the variable is unset or empty, protected
routes return 503 rather than allowing unauthenticated access.  Read-only
endpoints (GETs, /health) remain open — this is a personal dashboard.
"""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException

API_KEY_ENV = "AITRADER_API_KEY"
API_KEY_HEADER = "X-API-Key"


def require_api_key(
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency that guards mutating endpoints.

    Raises:
        HTTPException 503: if no API key is configured on the server
                           (fail closed, never open).
        HTTPException 401: if the header is missing or does not match.
    """
    configured = os.environ.get(API_KEY_ENV, "")
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="API key not configured – set AITRADER_API_KEY to enable "
            "mutating endpoints.",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def allowed_origins() -> list[str]:
    """Return CORS origins from ``ALLOWED_ORIGINS`` (comma-separated)."""
    raw = os.environ.get("ALLOWED_ORIGINS", "http://localhost:8006")
    return [origin.strip() for origin in raw.split(",") if origin.strip()]
