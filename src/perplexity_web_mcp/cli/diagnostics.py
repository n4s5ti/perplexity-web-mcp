"""Stable lifecycle checkpoints and actionable CLI error codes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
import os
import sys
import traceback
from typing import NoReturn


class CliCommand(StrEnum):
    """Commands with lifecycle checkpoint coverage."""

    ASK = "ask"
    COUNCIL = "council"
    LOGIN = "login"
    RESEARCH = "research"


class CliCheckpoint(StrEnum):
    """Allowlisted checkpoint phases that never contain user input."""

    START = "start"
    BROWSER_COOKIE_LOADED = "browser_cookie_loaded"
    SESSION_VALIDATED = "session_validated"
    TOKEN_SAVED = "token_saved"
    COMPLETE = "complete"
    FAILED = "failed"


class CliErrorCode(StrEnum):
    """Stable symbolic codes for actionable CLI failures."""

    INPUT_SOURCE_INVALID = "PWM_INPUT_SOURCE_INVALID"
    INPUT_MODEL_INVALID = "PWM_INPUT_MODEL_INVALID"
    INPUT_COUNCIL_INVALID = "PWM_INPUT_COUNCIL_INVALID"
    AUTH_REQUIRED = "PWM_AUTH_REQUIRED"
    AUTH_INVALID = "PWM_AUTH_INVALID"
    AUTH_FORBIDDEN = "PWM_AUTH_FORBIDDEN"
    AUTH_BROWSER_READ_FAILED = "PWM_AUTH_BROWSER_READ_FAILED"
    AUTH_SESSION_INVALID = "PWM_AUTH_SESSION_INVALID"
    AUTH_CANCELLED = "PWM_AUTH_CANCELLED"
    AUTH_SAVE_FAILED = "PWM_AUTH_SAVE_FAILED"
    QUERY_RATE_LIMITED = "PWM_QUERY_RATE_LIMITED"
    QUERY_FAILED = "PWM_QUERY_FAILED"
    INTERNAL_ERROR = "PWM_INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class _ErrorDetail:
    message: str
    remedy: str
    retryable: bool = False


_ERROR_DETAILS = {
    CliErrorCode.INPUT_SOURCE_INVALID: _ErrorDetail(
        "Unknown source.",
        "Run `pwm connectors list` or use a source shown by `pwm ask --help`.",
    ),
    CliErrorCode.INPUT_MODEL_INVALID: _ErrorDetail(
        "Unknown model.",
        "Choose a model shown by `pwm ask --help`.",
    ),
    CliErrorCode.INPUT_COUNCIL_INVALID: _ErrorDetail(
        "Unknown or incomplete council configuration.",
        "Review the model and member options shown by `pwm council --help`.",
    ),
    CliErrorCode.AUTH_REQUIRED: _ErrorDetail(
        "No usable Perplexity session is configured.",
        "Run `pwm login --from-browser` after signing in at perplexity.ai.",
    ),
    CliErrorCode.AUTH_INVALID: _ErrorDetail(
        "The saved session could not be validated.",
        "Check connectivity or sign in again, then run `pwm login --from-browser`.",
    ),
    CliErrorCode.AUTH_FORBIDDEN: _ErrorDetail(
        "Perplexity returned 403: the saved session is forbidden.",
        "Sign in again, then run `pwm login --from-browser`.",
    ),
    CliErrorCode.AUTH_BROWSER_READ_FAILED: _ErrorDetail(
        "The browser session could not be imported.",
        "Close the browser if needed, confirm it is signed in, and retry `pwm login --from-browser`.",
    ),
    CliErrorCode.AUTH_SESSION_INVALID: _ErrorDetail(
        "The imported browser session is expired or invalid.",
        "Sign in again at perplexity.ai, then retry the browser import.",
    ),
    CliErrorCode.AUTH_CANCELLED: _ErrorDetail(
        "Authentication was cancelled.",
        "Retry `pwm login` when ready.",
    ),
    CliErrorCode.AUTH_SAVE_FAILED: _ErrorDetail(
        "The validated session could not be saved securely.",
        "Check permissions for `~/.config/perplexity-web-mcp` and retry.",
    ),
    CliErrorCode.QUERY_RATE_LIMITED: _ErrorDetail(
        "Perplexity returned 429: the request was rate-limited.",
        "Run `pwm usage`, then retry after quota is available.",
        retryable=True,
    ),
    CliErrorCode.QUERY_FAILED: _ErrorDetail(
        "Perplexity could not complete the request.",
        "Check authentication and connectivity, then retry.",
        retryable=True,
    ),
    CliErrorCode.INTERNAL_ERROR: _ErrorDetail(
        "The CLI encountered an unexpected internal failure.",
        "Run `pwm doctor`; if it persists, capture the error code and report it.",
    ),
}


def emit_checkpoint(command: CliCommand, phase: CliCheckpoint, *, exit_code: int | None = None) -> None:
    """Write a fixed-field checkpoint to stderr without command arguments."""
    fields = ["pwm:", "event=checkpoint", f"command={command.value}", f"phase={phase.value}"]
    if exit_code is not None:
        fields.append(f"exit={exit_code}")
    sys.stderr.write(" ".join(fields) + "\n")


def emit_error(code: CliErrorCode) -> None:
    """Write a stable error record and fixed remediation without raw exception data."""
    detail = _ERROR_DETAILS[code]
    retryable = 1 if detail.retryable else 0
    sys.stderr.write(f"pwm: event=error code={code.value} retryable={retryable}\n")
    sys.stderr.write(f"pwm: {detail.message} {detail.remedy}\n")


def emit_debug_exception(error: Exception) -> None:
    """Print opt-in traceback details; default diagnostics remain redacted."""
    if os.environ.get("PWM_DEBUG", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    sys.stderr.write("pwm: debug=traceback warning=may-contain-sensitive-details\n")
    traceback.print_exception(error, file=sys.stderr)


def run_with_checkpoints(command: CliCommand, operation: Callable[[], int | None]) -> NoReturn:
    """Run a command, emit deterministic lifecycle checkpoints, and exit."""
    emit_checkpoint(command, CliCheckpoint.START)
    try:
        result = operation()
        exit_code = 0 if result is None else result
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except Exception as error:
        emit_error(CliErrorCode.INTERNAL_ERROR)
        emit_debug_exception(error)
        emit_checkpoint(command, CliCheckpoint.FAILED, exit_code=1)
        raise SystemExit(1) from None

    phase = CliCheckpoint.COMPLETE if exit_code == 0 else CliCheckpoint.FAILED
    emit_checkpoint(command, phase, exit_code=exit_code)
    raise SystemExit(exit_code)
