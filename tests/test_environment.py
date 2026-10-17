"""
tests/test_environment.py — does the part2 environment look right, and are the
safety rails on? Everything here is offline. Nothing here prints a secret.

Run from part2/ with the project interpreter:
    .venv\\Scripts\\python -m pytest
"""
from __future__ import annotations

import importlib
import os
import re
import socket
import subprocess
import sys
from pathlib import Path

import pytest

PART2 = Path(__file__).resolve().parents[1]

# pip name -> import name
REQUIRED_MODULES = {
    "requests": "requests",
    "pandas": "pandas",
    "python-dotenv": "dotenv",
    "pytest": "pytest",
    "mcp": "mcp",
    "openai": "openai",
    "anthropic": "anthropic",    # Claude as a second model provider (LLM_PROVIDER=anthropic), added 2026-10-17
    "pydantic": "pydantic",
    "gradio": "gradio",          # the chat window (app.py), added 2026-09-29
}

# The public settings template, exactly. None of these values is a secret.
EXPECTED_ENV_EXAMPLE = {
    "BRAPI_BASE_URL": "https://cassavabase.org/brapi/v2",
    "BRAPI_MODE": "offline",
    "BRAPI_SNAPSHOT_ID": "",
    "LLM_BASE_URL": "http://localhost:11434/v1",
    "LLM_API_KEY": "ollama",
    "LLM_MODEL": "",
    "LLM_ALLOW_REMOTE": "false",
    "LLM_REPLY_LIMIT_PARAM": "max_tokens",
    "LLM_REASONING_EFFORT": "",
}

EXPECTED_GITIGNORE_RULES = [
    ".venv/",
    ".env",
    ".env.*",
    "!.env.example",
    "cache/",
    "logs/",
    "out/",
    "snapshots/",
    "eval/results/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
]


def _parse_dotenv(text: str) -> dict[str, str]:
    """Tiny KEY=VALUE parser so this check needs no third-party package."""
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def _gitignore_rules() -> list[str]:
    lines = (PART2 / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


# --- 1. the interpreter -------------------------------------------------------

def test_python_version_is_supported():
    """mcp needs Python 3.10+. On failure the message names the interpreter."""
    assert sys.version_info >= (3, 10), f"Python {sys.version} at {sys.executable} is too old"


# --- 2. dependencies ----------------------------------------------------------

@pytest.mark.parametrize("pip_name,module", sorted(REQUIRED_MODULES.items()))
def test_required_package_is_importable(pip_name, module):
    try:
        importlib.import_module(module)
    except ImportError:
        pytest.fail(
            f"{pip_name} is not installed in this interpreter. "
            "Run: .venv\\Scripts\\python -m pip install -r requirements.txt"
        )


# --- 3. settings files --------------------------------------------------------

def test_real_env_file_is_never_read_by_tests():
    """If a real .env exists we only confirm git ignores it. Its contents are never read."""
    real_env = PART2 / ".env"
    if not real_env.exists():
        pytest.skip("no real .env present (fine)")
    assert ".env" in _gitignore_rules()


# --- 4. the safety rails themselves ---------------------------------------------

def test_pytest_config_excludes_live_and_model_by_default(pytestconfig):
    addopts = " ".join(pytestconfig.getini("addopts"))
    assert "not live and not model" in addopts
    registered = [m.split(":", 1)[0].strip() for m in pytestconfig.getini("markers")]
    assert "live" in registered and "model" in registered


def test_offline_settings_are_forced_during_tests():
    assert os.environ.get("BRAPI_MODE") == "offline"
    assert os.environ.get("LLM_ALLOW_REMOTE") == "false"


def test_proxies_are_neutralised_in_ordinary_tests():
    """A proxy on 127.0.0.1 would look local to the guard yet reach the internet."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert name not in os.environ, f"{name} is still set during an ordinary test"
    assert os.environ.get("NO_PROXY") == "*"
    assert os.environ.get("no_proxy") == "*"


def test_external_dns_lookup_is_blocked(network_blocked_error):
    with pytest.raises(network_blocked_error):
        socket.getaddrinfo("cassavabase.org", 443)


def test_external_tcp_connect_is_blocked(network_blocked_error):
    # 192.0.2.1 is a reserved documentation address (RFC 5737): never reachable,
    # numeric so no DNS lookup is involved. The guard must refuse it first.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        with pytest.raises(network_blocked_error):
            sock.connect(("192.0.2.1", 80))


def test_requests_to_the_brapi_server_is_blocked_in_ordinary_tests(network_blocked_error):
    requests = pytest.importorskip("requests")
    with pytest.raises(network_blocked_error):
        requests.get("https://cassavabase.org/brapi/v2/serverinfo", timeout=5)


def test_explicit_fakes_still_work(monkeypatch):
    """Blocking the network does not block fakes: a fake never opens a socket."""
    requests = pytest.importorskip("requests")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"result": {"data": ["cassava"]}}

    monkeypatch.setattr(requests, "get", lambda url, **kwargs: FakeResponse())
    reply = requests.get("https://cassavabase.org/brapi/v2/commoncropnames")
    assert reply.json()["result"]["data"] == ["cassava"]


def test_loopback_sockets_still_work():
    """Local-only sockets must keep working (asyncio on Windows depends on them)."""
    left, right = socket.socketpair()
    left.close()
    right.close()


# --- 5. subprocesses ------------------------------------------------------------

def test_subprocess_gets_the_offline_configuration(offline_env):
    assert "HTTPS_PROXY" not in offline_env and "https_proxy" not in offline_env
    code = (
        "import os; print(os.environ['BRAPI_MODE'], os.environ['LLM_ALLOW_REMOTE'], "
        "os.environ['NO_PROXY'])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=offline_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.split() == ["offline", "false", "*"]
