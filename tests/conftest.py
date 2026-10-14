"""
tests/conftest.py — safety rails that pytest loads automatically for every test.

What it does, in plain words:

1. Forces the OFFLINE configuration for every test: BRAPI_MODE=offline and
   LLM_ALLOW_REMOTE=false are put into the environment, so code under test —
   and any subprocess it starts — sees the offline settings.

2. Blocks external network access in ordinary tests. "External" means any
   address that is not this computer. It patches the low-level socket module
   (what every HTTP library uses underneath), so a slip in any library is
   caught, not just `requests`. A blocked attempt raises NetworkBlockedError
   with a plain message.

   It also removes proxy variables (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY) and
   sets NO_PROXY=* for the test. Why: a proxy running on this computer
   (127.0.0.1) would be "local" to the socket guard, yet it forwards traffic
   to the internet. With proxies off, libraries connect directly and the
   guard sees the real destination.

   Tests marked @pytest.mark.live or @pytest.mark.model are NOT blocked
   (they are excluded by default in pytest.ini anyway).

3. Provides the `offline_env` fixture: an environment dict for tests that
   start a subprocess, with the offline settings forced.

Explicit test fakes keep working. A fake client object, `monkeypatch.setattr`
on `requests.get`, the `responses` library, etc. never open a real socket, so
the guard never sees them.

Loopback (127.0.0.1 / ::1 / localhost) and Unix sockets stay allowed: they
never leave this computer, and Python's asyncio on Windows creates a loopback
socket pair internally — blocking it would break every asyncio test.

Nothing in this file prints an environment value.
"""
from __future__ import annotations

import ipaddress
import os
import socket

import pytest

# Set before test-module collection; inherited by all offline subprocesses.
# No test needs to load the developer's real local secrets file.
os.environ["PART2_IGNORE_DOTENV"] = "true"

# Settings every test runs under. Tests may override with monkeypatch.setenv.
OFFLINE_SETTINGS = {
    "PART2_IGNORE_DOTENV": "true",
    "LLM_API_KEY": "offline-test-placeholder",
    "BRAPI_MODE": "offline",
    "LLM_ALLOW_REMOTE": "false",
}

# Proxy variables removed in ordinary tests, and the "bypass every proxy" setting.
PROXY_VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")
NO_PROXY_SETTINGS = {"NO_PROXY": "*", "no_proxy": "*"}

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


class NetworkBlockedError(RuntimeError):
    """Raised when an ordinary test tries to reach an address outside this computer."""


def _is_local(host) -> bool:
    """True for a host that never leaves this computer (loopback, wildcard, Unix path)."""
    if host is None or host == "" or host == b"":
        return True  # wildcard / bind-only; no outbound traffic
    if isinstance(host, bytes):
        host = host.decode(errors="ignore")
    if not isinstance(host, str):
        return False
    host = host.strip("[]").split("%", 1)[0]  # drop IPv6 brackets and scope id
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False  # a real host name would need DNS -> external


def _address_host(address):
    """Pull the host part out of a socket address of any family."""
    if isinstance(address, (str, bytes)):
        return None  # AF_UNIX path: local by definition
    if isinstance(address, tuple) and address:
        return address[0]
    return address


def _install_network_guard(monkeypatch: pytest.MonkeyPatch, test_id: str) -> None:
    """Patch the socket module so anything non-local raises NetworkBlockedError."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex

    def refuse(what: str, host) -> None:
        raise NetworkBlockedError(
            f"{test_id}: blocked {what} to {host!r}. Ordinary tests run offline. "
            "Use a fake, or mark the test @pytest.mark.live / @pytest.mark.model."
        )

    def guarded_connect(self, address):
        host = _address_host(address)
        if not _is_local(host):
            refuse("connect", host)
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        host = _address_host(address)
        if not _is_local(host):
            refuse("connect_ex", host)
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(host, port, *args, **kwargs):
        if not _is_local(host):
            refuse("DNS lookup", host)
        return real_getaddrinfo(host, port, *args, **kwargs)

    def guarded_gethostbyname(name):
        if not _is_local(name):
            refuse("DNS lookup", name)
        return real_gethostbyname(name)

    def guarded_gethostbyname_ex(name):
        if not _is_local(name):
            refuse("DNS lookup", name)
        return real_gethostbyname_ex(name)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    monkeypatch.setattr(socket, "gethostbyname", guarded_gethostbyname)
    monkeypatch.setattr(socket, "gethostbyname_ex", guarded_gethostbyname_ex)


def _neutralise_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove proxy variables and tell every library to bypass proxies for all hosts."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for key, value in NO_PROXY_SETTINGS.items():
        monkeypatch.setenv(key, value)


@pytest.fixture(autouse=True)
def _offline_guard(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """Runs around EVERY test: offline settings always; proxies off + network block unless live/model.

    monkeypatch undoes the environment changes and the socket patch after each test.
    """
    for key, value in OFFLINE_SETTINGS.items():
        monkeypatch.setenv(key, value)
    is_live = request.node.get_closest_marker("live") is not None
    is_model = request.node.get_closest_marker("model") is not None
    if not (is_live or is_model):
        _neutralise_proxies(monkeypatch)
        _install_network_guard(monkeypatch, request.node.nodeid)
    yield


@pytest.fixture
def offline_env() -> dict[str, str]:
    """Environment for subprocess tests: a copy of os.environ with the offline settings
    forced and proxies neutralised.

    Usage:
        subprocess.run([sys.executable, "-m", "something"], env=offline_env, ...)
    """
    env = dict(os.environ)
    for name in PROXY_VARIABLES:
        env.pop(name, None)
    env.update(OFFLINE_SETTINGS)
    env.update(NO_PROXY_SETTINGS)
    return env


@pytest.fixture
def network_blocked_error() -> type[NetworkBlockedError]:
    """The exception class the guard raises — for tests that prove the guard works."""
    return NetworkBlockedError
