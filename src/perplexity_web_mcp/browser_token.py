"""Read a Perplexity session cookie from a supported browser profile."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .auth import session_token_from_cookie_pairs


BROWSER_LOADERS = {
    "chrome": "chrome",
    "chromium": "chromium",
    "brave": "brave",
    "edge": "edge",
    "firefox": "firefox",
    "librewolf": "librewolf",
    "opera": "opera",
    "opera-gx": "opera_gx",
    "vivaldi": "vivaldi",
}
"""CLI browser names mapped to browser_cookie3 loader attributes."""

SUPPORTED_BROWSERS = tuple(BROWSER_LOADERS)
CHROMIUM_BROWSERS = frozenset({"chrome", "chromium", "brave", "edge", "opera", "opera-gx", "vivaldi"})


class BrowserTokenError(ValueError):
    """Raised when a Perplexity browser session cannot be imported."""


def load_browser_token(
    browser: str = "auto", cookie_file: Path | None = None, *, cookie_module: Any | None = None
) -> str:
    """Return the Perplexity session token from a browser cookie store.

    ``cookie_module`` is injectable so callers can test extraction without
    accessing a real browser profile.
    """
    browser = browser.lower()
    if browser != "auto" and browser not in BROWSER_LOADERS:
        supported = ", ".join(("auto", *SUPPORTED_BROWSERS))
        raise BrowserTokenError(f"Unsupported browser. Choose one of: {supported}.")

    if cookie_file is not None and not cookie_file.is_file():
        raise BrowserTokenError("Cookie file is unavailable. Check --cookie-file and try again.")

    if cookie_module is None:
        try:
            import browser_cookie3 as cookie_module
        except ImportError as error:
            raise BrowserTokenError("Browser cookie support is unavailable. Reinstall perplexity-web-mcp.") from error

    candidates = SUPPORTED_BROWSERS if browser == "auto" else (browser,)
    if cookie_file is not None and browser == "auto":
        candidates = tuple(name for name in candidates if name in CHROMIUM_BROWSERS)
    elif cookie_file is not None and browser not in CHROMIUM_BROWSERS:
        raise BrowserTokenError("--cookie-file is supported only for Chromium-family browsers.")

    loader_errors = 0
    for browser_name in candidates:
        loader = getattr(cookie_module, BROWSER_LOADERS[browser_name], None)
        if not callable(loader):
            if browser != "auto":
                raise BrowserTokenError("The installed browser cookie library does not support the requested browser.")
            continue

        kwargs: dict[str, str] = {"domain_name": "perplexity.ai"}
        if cookie_file is not None:
            kwargs["cookie_file"] = str(cookie_file)

        try:
            cookies = loader(**kwargs)
        except Exception:
            loader_errors += 1
            continue

        token = session_token_from_cookie_pairs(
            (name, value)
            for cookie in cookies
            if isinstance(name := getattr(cookie, "name", None), str)
            and isinstance(value := getattr(cookie, "value", None), str)
        )
        if token:
            return token

    if browser != "auto" and loader_errors:
        raise BrowserTokenError("Unable to read cookies from the selected browser. Close it and try again.")

    raise BrowserTokenError(
        "No signed-in Perplexity session was found. Sign in at perplexity.ai in a supported browser and try again."
    )
