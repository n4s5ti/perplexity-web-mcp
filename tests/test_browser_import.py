"""Focused tests for importing a Perplexity browser session without disclosing it."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner
import pytest

from perplexity_web_mcp.browser_token import SUPPORTED_BROWSERS, BrowserTokenError, load_browser_token
from perplexity_web_mcp.cli.auth import (
    SessionValidationResult,
    SessionValidationStatus,
    SubscriptionTier,
    UserInfo,
    auth_non_interactive,
    import_browser_session,
)
from perplexity_web_mcp.cli.main import cli
from perplexity_web_mcp.constants import SESSION_COOKIE_NAME


TEST_TOKEN = "browser-session-token-must-never-appear-in-output-7f3a9c"


def authenticated_user() -> UserInfo:
    """Return a representative authenticated account without making a request."""
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


def assert_token_is_not_disclosed(output: str) -> None:
    """Assert neither the raw token nor recognizable fragments are emitted."""
    assert TEST_TOKEN not in output
    assert TEST_TOKEN[:16] not in output
    assert TEST_TOKEN[-16:] not in output


class TestBrowserTokenExtraction:
    """Tests for deterministic, injected browser-cookie extraction."""

    def test_supported_browser_names_cover_documented_browsers(self) -> None:
        assert set(SUPPORTED_BROWSERS) >= {
            "chrome",
            "chromium",
            "brave",
            "edge",
            "firefox",
            "librewolf",
            "opera",
            "opera-gx",
            "vivaldi",
        }

    @pytest.mark.parametrize(
        ("browser", "loader_name"),
        [
            ("chrome", "chrome"),
            ("chromium", "chromium"),
            ("brave", "brave"),
            ("edge", "edge"),
            ("firefox", "firefox"),
            ("librewolf", "librewolf"),
            ("opera", "opera"),
            ("opera-gx", "opera_gx"),
            ("vivaldi", "vivaldi"),
        ],
    )
    def test_explicit_browser_extracts_only_the_session_cookie(self, browser: str, loader_name: str) -> None:
        loader = MagicMock(
            return_value=[
                SimpleNamespace(name="unrelated-cookie", value="ignore-me"),
                SimpleNamespace(name=SESSION_COOKIE_NAME, value=TEST_TOKEN),
            ]
        )
        cookie_module = SimpleNamespace(**{loader_name: loader})

        assert load_browser_token(browser, cookie_module=cookie_module) == TEST_TOKEN
        loader.assert_called_once_with(domain_name="perplexity.ai")

    def test_auto_browser_uses_first_available_loader_with_session_cookie(self) -> None:
        chrome = MagicMock(return_value=[SimpleNamespace(name=SESSION_COOKIE_NAME, value=TEST_TOKEN)])

        assert load_browser_token(cookie_module=SimpleNamespace(chrome=chrome)) == TEST_TOKEN
        chrome.assert_called_once_with(domain_name="perplexity.ai")

    def test_browser_reassembles_session_cookie_chunks_in_numeric_order(self) -> None:
        firefox = MagicMock(
            return_value=[
                SimpleNamespace(name=f"{SESSION_COOKIE_NAME}.1", value=TEST_TOKEN[24:]),
                SimpleNamespace(name=f"{SESSION_COOKIE_NAME}.0", value=TEST_TOKEN[:24]),
            ]
        )

        assert load_browser_token("firefox", cookie_module=SimpleNamespace(firefox=firefox)) == TEST_TOKEN

    def test_chromium_cookie_file_is_passed_to_browser_loader(self, tmp_path: Path) -> None:
        cookie_file = tmp_path / "Cookies"
        cookie_file.touch()
        chrome = MagicMock(return_value=[SimpleNamespace(name=SESSION_COOKIE_NAME, value=TEST_TOKEN)])

        assert load_browser_token("chrome", cookie_file, cookie_module=SimpleNamespace(chrome=chrome)) == TEST_TOKEN
        assert Path(chrome.call_args.kwargs["cookie_file"]) == cookie_file
        assert chrome.call_args.kwargs["domain_name"] == "perplexity.ai"

    def test_missing_session_cookie_raises_browser_token_error(self) -> None:
        chrome = MagicMock(return_value=[SimpleNamespace(name="next-auth.session-token", value=TEST_TOKEN)])

        with pytest.raises(BrowserTokenError):
            load_browser_token("chrome", cookie_module=SimpleNamespace(chrome=chrome))

    def test_loader_failure_raises_browser_token_error_without_cookie_value(self) -> None:
        chrome = MagicMock(side_effect=OSError("cookie database is locked"))

        with pytest.raises(BrowserTokenError) as error:
            load_browser_token("chrome", cookie_module=SimpleNamespace(chrome=chrome))

        assert TEST_TOKEN not in str(error.value)


class TestBrowserSessionImport:
    """Tests for validation, saving, and safe browser-login output."""

    def test_browser_import_validates_account_before_saving(self) -> None:
        calls: list[str] = []

        def validate_user_session(token: str) -> SessionValidationResult:
            assert token == TEST_TOKEN
            calls.append("validate")
            return SessionValidationResult(SessionValidationStatus.VALID, authenticated_user())

        def save_token(token: str) -> bool:
            assert token == TEST_TOKEN
            calls.append("save")
            return True

        with (
            patch("perplexity_web_mcp.cli.auth.load_browser_token", return_value=TEST_TOKEN),
            patch("perplexity_web_mcp.cli.auth.validate_user_session", side_effect=validate_user_session),
            patch("perplexity_web_mcp.cli.auth.save_token_to_config", side_effect=save_token),
        ):
            assert import_browser_session("auto", None, auto_save=True) is True

        assert calls == ["validate", "save"]

    def test_browser_import_rejects_invalid_token_without_saving(self) -> None:
        save_token = MagicMock(return_value=True)

        with (
            patch("perplexity_web_mcp.cli.auth.load_browser_token", return_value=TEST_TOKEN),
            patch(
                "perplexity_web_mcp.cli.auth.validate_user_session",
                return_value=SessionValidationResult(SessionValidationStatus.REJECTED),
            ),
            patch("perplexity_web_mcp.cli.auth.save_token_to_config", save_token),
        ):
            assert import_browser_session("auto", None, auto_save=True) is False

        save_token.assert_not_called()

    def test_browser_import_no_save_validates_without_writing_token(self) -> None:
        save_token = MagicMock(return_value=True)

        with (
            patch("perplexity_web_mcp.cli.auth.load_browser_token", return_value=TEST_TOKEN),
            patch(
                "perplexity_web_mcp.cli.auth.validate_user_session",
                return_value=SessionValidationResult(SessionValidationStatus.VALID, authenticated_user()),
            ),
            patch("perplexity_web_mcp.cli.auth.save_token_to_config", save_token),
        ):
            assert import_browser_session("auto", None, auto_save=False) is True

        save_token.assert_not_called()

    def test_browser_import_output_does_not_disclose_session_token(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("perplexity_web_mcp.cli.auth.load_browser_token", return_value=TEST_TOKEN),
            patch(
                "perplexity_web_mcp.cli.auth.validate_user_session",
                return_value=SessionValidationResult(SessionValidationStatus.VALID, authenticated_user()),
            ),
            patch("perplexity_web_mcp.cli.auth.save_token_to_config", return_value=True),
        ):
            assert import_browser_session("auto", None, auto_save=True) is True

        captured = capsys.readouterr()
        assert_token_is_not_disclosed(captured.out + captured.err)


class TestLoginBrowserImportWiring:
    """Tests for forwarding browser-import options through the shared Click CLI."""

    def test_login_forwards_browser_import_options_to_auth_cli(self, tmp_path: Path) -> None:
        cookie_file = tmp_path / "Cookies"
        captured_argv: list[str] = []

        def auth_main() -> None:
            captured_argv.extend(sys.argv)

        with patch("perplexity_web_mcp.cli.auth.main", side_effect=auth_main):
            result = CliRunner().invoke(
                cli,
                ["login", "--from-browser", "--browser", "brave", "--cookie-file", str(cookie_file), "--no-save"],
            )

        assert result.exit_code == 0
        assert captured_argv == [
            "pwm-auth",
            "--from-browser",
            "--browser",
            "brave",
            "--cookie-file",
            str(cookie_file),
            "--no-save",
        ]

    def test_login_forwards_totp_code_to_auth_cli(self) -> None:
        captured_argv: list[str] = []

        def auth_main() -> None:
            captured_argv.extend(sys.argv)

        with patch("perplexity_web_mcp.cli.auth.main", side_effect=auth_main):
            result = CliRunner().invoke(
                cli,
                ["login", "--email", "reader@example.com", "--code", "123456", "--totp-code", "654321"],
            )

        assert result.exit_code == 0
        assert captured_argv == [
            "pwm-auth",
            "--email",
            "reader@example.com",
            "--code",
            "123456",
            "--totp-code",
            "654321",
        ]


class TestOtpTokenDisclosure:
    """Tests that the existing OTP completion flow keeps the token out of output."""

    def test_otp_completion_output_does_not_disclose_session_token(self, capsys: pytest.CaptureFixture[str]) -> None:
        with (
            patch("perplexity_web_mcp.cli.auth._initialize_session", return_value=(MagicMock(), "csrf")),
            patch("perplexity_web_mcp.cli.auth._validate_and_get_redirect_url", return_value="https://callback"),
            patch("perplexity_web_mcp.cli.auth._complete_auth_callback", return_value=TEST_TOKEN),
            patch(
                "perplexity_web_mcp.cli.auth.validate_user_session",
                return_value=SessionValidationResult(SessionValidationStatus.VALID, authenticated_user()),
            ),
            patch("perplexity_web_mcp.cli.auth.save_token_to_config", return_value=True),
        ):
            assert auth_non_interactive("reader@example.com", code="123456") == TEST_TOKEN

        captured = capsys.readouterr()
        assert_token_is_not_disclosed(captured.out + captured.err)
