"""
tests/test_tool_failure_log.py - brapi_mcp_server._map_exception called an undefined `log` (the status doc's "other open", from
2026-10-22; coach 2026-10-23 11:50 UTC, NEXT 2): a tool that failed in an unexpected way raised NameError out of dispatch,
which promises to ALWAYS return a ToolResult, and its traceback reached no log. SYNTHETIC context. No network, no model.
"""
from __future__ import annotations

import dataclasses
import logging

import brapi_mcp_server as srv
from part2_support import synthetic_context


def test_an_unexpected_tool_failure_is_a_coded_result_and_its_details_go_to_the_log(tmp_path, monkeypatch, caplog):
    ctx = synthetic_context(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out", run_id="run_tool_failure")

    def boom(_ctx, _args):
        raise RuntimeError("an unexpected failure inside the tool")

    monkeypatch.setitem(srv.TOOLS, "server_info", dataclasses.replace(srv.TOOLS["server_info"], fn=boom))
    with caplog.at_level(logging.ERROR):
        r = srv.dispatch(ctx, "server_info", {})
    assert not r.ok and r.error.code == "internal_error" and r.error.message.startswith("RuntimeError")
    assert "an unexpected failure" not in r.error.message                 # the details stay in the server log, not in the message
    assert any(record.exc_info and "an unexpected failure inside the tool" in str(record.exc_info[1]) for record in caplog.records)
