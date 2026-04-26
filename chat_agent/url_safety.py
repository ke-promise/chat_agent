"""Public web URL validation helpers."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse


class URLSafetyError(ValueError):
    """Raised when a URL is not safe for outbound web fetching."""


def ensure_public_http_url(url: str) -> str:
    """Validate that a URL targets a public http/https endpoint."""
    text = str(url or "").strip()
    if not text:
        raise URLSafetyError("url is required")

    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"}:
        raise URLSafetyError("only http:// and https:// URLs are allowed")
    if not parsed.netloc:
        raise URLSafetyError("URL must include a hostname")

    host = parsed.hostname
    if not host:
        raise URLSafetyError("URL must include a hostname")
    if host.lower() == "localhost":
        raise URLSafetyError("localhost URLs are not allowed")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None

    if address and (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise URLSafetyError("private or loopback IP URLs are not allowed")

    return parsed.geturl()
