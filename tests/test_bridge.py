"""
tests/test_bridge.py — one McpTools session gives the agents the same tools over MCP
(a real stdio child) and directly (in-process), decodes replies deliberately, and
never leaves a child process behind — not after a timeout, not after cancellation.

Everything is offline on the SYNTHETIC fixture; the child inherits the offline env.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import anyio
import pytest
from mcp import types

import brapi_mcp_server as srv
import mcp_bridge as mb
from mcp_bridge import BridgeClosed, BridgeError, BridgeTimeout, McpTools, decode_call_result, to_model_tools
from part2_support import synthetic_context

PART2 = Path(__file__).resolve().parents[1]


def server_args(tmp_path, run_id="run_bridge"):
    return ["--offline", "--fixture", "synthetic", "--cache-dir", str(tmp_path / "cache"),
            "--out-dir", str(tmp_path / "out"), "--run-id", run_id]


def run(fn, *args):
    return anyio.run(fn, *args)


@pytest.fixture
def captured_processes(monkeypatch):
    """Record every child process the MCP client starts, so tests can prove it was closed."""
    import mcp.client.stdio as stdio_mod

    real = stdio_mod._create_platform_compatible_process
    processes = []

    async def wrapper(*args, **kwargs):
        process = await real(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stdio_mod, "_create_platform_compatible_process", wrapper)
    return processes


def is_finished(process) -> bool:
    popen = getattr(process, "popen", None)                 # Windows wrapper
    if popen is not None:
        return popen.poll() is not None
    return getattr(process, "returncode", None) is not None  # anyio Process


# --- 1. the server child: how it is started ------------------------------------------------

# --- 2. identical schemas over both routes, converted once ----------------------------------

def test_model_tools_rejects_malformed_schemas():
    with pytest.raises(BridgeError):
        to_model_tools([{"name": "x"}])                    # no inputSchema
    with pytest.raises(BridgeError):
        to_model_tools([{"inputSchema": {}}])              # no name


# --- 3. equivalent results over both routes -------------------------------------------------

# --- 4. deliberate decoding ---------------------------------------------------------------------

def _reply(*, content=None, structured=None, is_error=False):
    return SimpleNamespace(content=list(content or []), structuredContent=structured, isError=is_error)


def test_decode_prefers_structured_content_and_accepts_text_toolresults():
    payload = json.loads(srv.dump_json(srv.ToolResult(ok=True, data={"n": 1}, complete=True)))
    decoded = decode_call_result(_reply(content=[types.TextContent(type="text", text=json.dumps(payload))], structured=payload), "t")
    assert decoded.ok and decoded.data == {"n": 1}
    text_only = decode_call_result(_reply(content=[types.TextContent(type="text", text=json.dumps(payload))]), "t")
    assert text_only == decoded


def test_decode_turns_mcp_level_errors_into_coded_results():
    decoded = decode_call_result(_reply(content=[types.TextContent(type="text", text="Input validation error: 5 is not of type 'string'")], is_error=True), "get_study")
    assert not decoded.ok and decoded.error.code == "invalid_argument" and "not of type" in decoded.error.message


@pytest.mark.parametrize("reply,fragment", [
    (_reply(content=[]), "empty reply"),
    (_reply(content=[types.ImageContent(type="image", data="AAAA", mimeType="image/png")]), "unsupported content"),
    (_reply(content=[types.TextContent(type="text", text="{not json")]), "neither a ToolResult"),
    (_reply(content=[types.TextContent(type="text", text='{"hello": "world"}')]), "neither a ToolResult"),
    (_reply(structured={"ok": True, "data": {"x": 1}, "bogus_field": 1}), "not a ToolResult"),
])
def test_decode_rejects_malformed_or_unsupported_content(reply, fragment):
    with pytest.raises(BridgeError) as info:
        decode_call_result(reply, "some_tool")
    assert fragment in str(info.value)


# --- 5. lifetime: closed sessions, timeouts, cancellation, no orphans ------------------------------

def test_calls_outside_the_context_are_refused(tmp_path):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_c")
    bridge = McpTools.direct(ctx)

    async def before():
        await bridge.call("server_info", {})

    with pytest.raises(BridgeClosed):
        run(before)

    async def after():
        async with bridge:
            pass
        await bridge.schemas()

    with pytest.raises(BridgeClosed):
        run(after)


class _ClosingStack:
    """Stands in for the exit stack: closing it raises what a late child reply would raise."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    async def aclose(self) -> None:
        raise self.exc


def _close_bridge_with(tmp_path, exc: BaseException) -> None:
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_close")
    bridge = McpTools.direct(ctx)
    bridge._open, bridge._stack = True, _ClosingStack(exc)

    async def close():
        await bridge.__aexit__(None, None, None)

    run(close)
    assert bridge._stack is None and not bridge._open


def test_teardown_swallows_only_stream_closing_noise(tmp_path):
    noise = ExceptionGroup("unhandled errors in a TaskGroup", [anyio.BrokenResourceError()])
    _close_bridge_with(tmp_path, noise)                                             # flat, as anyio raises it
    _close_bridge_with(tmp_path, ExceptionGroup("outer", [noise, ExceptionGroup("g", [anyio.ClosedResourceError()])]))   # nested
    for real in (ExceptionGroup("x", [RuntimeError("real problem")]),
                 ExceptionGroup("mixed", [anyio.BrokenResourceError(), RuntimeError("real problem")])):
        with pytest.raises(BaseExceptionGroup) as info:
            _close_bridge_with(tmp_path, real)
        assert info.value.subgroup(RuntimeError) is not None                          # the real problem is kept
