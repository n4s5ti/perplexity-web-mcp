"""Behavioral contracts for CLI checkpoints and stable error codes."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import pytest

from perplexity_web_mcp.browser_token import BrowserTokenError
from perplexity_web_mcp.cli.auth import (
    SessionValidationResult,
    SessionValidationStatus,
    SubscriptionTier,
    UserInfo,
    import_browser_session,
    validate_user_session,
)
from perplexity_web_mcp.cli.auth import main as auth_main
from perplexity_web_mcp.cli.diagnostics import CliErrorCode
from perplexity_web_mcp.cli.main import cli
from perplexity_web_mcp.exceptions import AuthenticationError, RateLimitError


SECRET_QUERY = "private acquisition target 7f4e"
SECRET_TOKEN = "header.payload.signature"


def _authenticated_user() -> UserInfo:
    return UserInfo(
        id="user-id",
        email="reader@example.com",
        username="reader",
        name=None,
        subscription_tier=SubscriptionTier.PRO,
        subscription_status="active",
        subscription_source="stripe",
        payment_tier="pro",
        is_in_organization=False,
    )


class TestQueryCheckpoints:
    """Query commands preserve stdout while emitting fixed-field stderr checkpoints."""

    def test_ask_success_preserves_stdout_without_logging_query(self) -> None:
        with patch("perplexity_web_mcp.cli.main.ask", return_value="answer"):
            result = CliRunner().invoke(cli, ["ask", SECRET_QUERY, "--model", "sonar"])

        assert result.exit_code == 0
        assert result.stdout == "answer\n"
        assert "event=checkpoint command=ask phase=start" in result.stderr
        assert "event=checkpoint command=ask phase=complete exit=0" in result.stderr
        assert SECRET_QUERY not in result.stderr

    @pytest.mark.parametrize(
        ("error", "code"),
        [
            (AuthenticationError(), CliErrorCode.AUTH_FORBIDDEN),
            (RateLimitError(), CliErrorCode.QUERY_RATE_LIMITED),
            (RuntimeError(SECRET_TOKEN), CliErrorCode.INTERNAL_ERROR),
        ],
    )
    def test_ask_failures_return_stable_codes_without_raw_details(self, error: Exception, code: CliErrorCode) -> None:
        with patch("perplexity_web_mcp.cli.main.ask", side_effect=error):
            result = CliRunner().invoke(cli, ["ask", SECRET_QUERY, "--model", "sonar"])

        assert result.exit_code == 1
        assert f"code={code.value}" in result.stderr
        assert "event=checkpoint command=ask phase=failed exit=1" in result.stderr
        assert SECRET_QUERY not in result.stderr
        assert SECRET_TOKEN not in result.stderr

    def test_internal_error_traceback_is_explicitly_opt_in(self) -> None:
        with patch("perplexity_web_mcp.cli.main.ask", side_effect=RuntimeError(SECRET_TOKEN)):
            result = CliRunner().invoke(
                cli,
                ["ask", SECRET_QUERY, "--model", "sonar"],
                env={"PWM_DEBUG": "1"},
            )

        assert result.exit_code == 1
        assert f"code={CliErrorCode.INTERNAL_ERROR.value}" in result.stderr
        assert "debug=traceback warning=may-contain-sensitive-details" in result.stderr
        assert "RuntimeError" in result.stderr
        assert SECRET_TOKEN in result.stderr
        assert SECRET_QUERY not in result.stderr

    def test_research_rate_limit_has_command_specific_failed_checkpoint(self) -> None:
        with patch("perplexity_web_mcp.cli.main.ask", side_effect=RateLimitError()):
            result = CliRunner().invoke(cli, ["research", SECRET_QUERY])

        assert result.exit_code == 1
        assert f"code={CliErrorCode.QUERY_RATE_LIMITED.value}" in result.stderr
        assert "event=checkpoint command=research phase=failed exit=1" in result.stderr
        assert SECRET_QUERY not in result.stderr

    def test_council_input_failure_has_stable_code_and_no_query_leak(self) -> None:
        result = CliRunner().invoke(cli, ["council", SECRET_QUERY, "--models", "sonar"])

        assert result.exit_code == 1
        assert f"code={CliErrorCode.INPUT_COUNCIL_INVALID.value}" in result.stderr
        assert "event=checkpoint command=council phase=failed exit=1" in result.stderr
        assert SECRET_QUERY not in result.stderr


class TestSessionValidationOutcomes:
    """HTTP rejection and transient validation failures remain distinguishable."""

    @pytest.mark.parametrize("status_code", [401, 403])
    def test_http_auth_rejection_is_typed(self, status_code: int) -> None:
        session = MagicMock()
        session.get.return_value.status_code = status_code
        with patch("perplexity_web_mcp.cli.auth.Session") as session_type:
            session_type.return_value.__enter__.return_value = session
            result = validate_user_session(SECRET_TOKEN)

        assert result.status is SessionValidationStatus.REJECTED
        assert result.user_info is None
        assert result.error is None

    def test_service_failure_is_typed_as_retryable_unavailable(self) -> None:
        session = MagicMock()
        session.get.return_value.status_code = 503
        with patch("perplexity_web_mcp.cli.auth.Session") as session_type:
            session_type.return_value.__enter__.return_value = session
            result = validate_user_session(SECRET_TOKEN)

        assert result.status is SessionValidationStatus.UNAVAILABLE
        assert result.user_info is None
        assert isinstance(result.error, RuntimeError)

    @pytest.mark.parametrize(
        ("validation", "code", "retryable"),
        [
            (
                SessionValidationResult(SessionValidationStatus.REJECTED),
                CliErrorCode.AUTH_SESSION_INVALID,
                0,
            ),
            (
                SessionValidationResult(SessionValidationStatus.UNAVAILABLE, error=RuntimeError(SECRET_TOKEN)),
                CliErrorCode.AUTH_VALIDATION_UNAVAILABLE,
                1,
            ),
        ],
    )
    def test_browser_import_maps_validation_outcome_without_secret_leakage(
        self,
        validation: SessionValidationResult,
        code: CliErrorCode,
        retryable: int,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        with (
            patch("perplexity_web_mcp.cli.auth.load_browser_token", return_value=SECRET_TOKEN),
            patch("perplexity_web_mcp.cli.auth.validate_user_session", return_value=validation),
        ):
            assert import_browser_session("firefox") is False

        captured = capsys.readouterr()
        assert f"code={code.value} retryable={retryable}" in captured.err
        assert SECRET_TOKEN not in captured.err


class TestLoginCheckpoints:
    """Browser login exposes lifecycle state without exposing credential material."""

    def test_browser_import_emits_validation_and_save_checkpoints(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("perplexity_web_mcp.cli.auth.load_browser_token", return_value=SECRET_TOKEN),
            patch(
                "perplexity_web_mcp.cli.auth.validate_user_session",
                return_value=SessionValidationResult(SessionValidationStatus.VALID, _authenticated_user()),
            ),
            patch("perplexity_web_mcp.cli.auth.save_token_to_config", return_value=True),
        ):
            assert import_browser_session("firefox") is True

        captured = capsys.readouterr()
        assert "phase=browser_cookie_loaded" in captured.err
        assert "phase=session_validated" in captured.err
        assert "phase=token_saved" in captured.err
        assert SECRET_TOKEN not in captured.err

    def test_browser_loader_failure_has_stable_code_without_exception_details(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with patch(
            "perplexity_web_mcp.cli.auth.load_browser_token",
            side_effect=BrowserTokenError(SECRET_TOKEN),
        ):
            assert import_browser_session("firefox") is False

        captured = capsys.readouterr()
        assert f"code={CliErrorCode.AUTH_BROWSER_READ_FAILED.value}" in captured.err
        assert SECRET_TOKEN not in captured.err

    def test_login_wrapper_records_terminal_exit_without_logging_email_or_code(self) -> None:
        with patch("perplexity_web_mcp.cli.auth.main", side_effect=SystemExit(1)):
            result = CliRunner().invoke(
                cli,
                ["login", "--email", "private@example.com", "--code", "123456"],
            )

        assert result.exit_code == 1
        assert "event=checkpoint command=login phase=start" in result.stderr
        assert "event=checkpoint command=login phase=failed exit=1" in result.stderr
        assert "private@example.com" not in result.stderr
        assert "123456" not in result.stderr

    def test_interactive_cancellation_is_not_reported_as_success(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch.object(sys, "argv", ["pwm-auth"]),
            patch("perplexity_web_mcp.cli.auth._show_header", side_effect=KeyboardInterrupt),
            pytest.raises(SystemExit) as error,
        ):
            auth_main()

        assert error.value.code == 130
        assert f"code={CliErrorCode.AUTH_CANCELLED.value}" in capsys.readouterr().err

    def test_login_route_preserves_cancel_exit_and_failed_checkpoint(self) -> None:
        with patch("perplexity_web_mcp.cli.auth._show_header", side_effect=KeyboardInterrupt):
            result = CliRunner().invoke(cli, ["login"])

        assert result.exit_code == 130
        assert f"code={CliErrorCode.AUTH_CANCELLED.value}" in result.stderr
        assert "event=checkpoint command=login phase=failed exit=130" in result.stderr
        assert "phase=complete" not in result.stderr
