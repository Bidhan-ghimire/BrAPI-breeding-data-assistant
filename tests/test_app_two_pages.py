"""
tests/test_app_two_pages.py - the app's own two pages per fetch (Bidhan, 2026-10-23, item 14), through a whole chat: a study whose
observations and plot list each span two pages is read in full at the app's DEFAULT setting (the review of item 14 found that no
test sent a second page through the tools: paging only above two pages passed every test). The fixtures of
test_local_mcp_controller.py; only the fake server differs. Fake server and scripted models only.
"""
from __future__ import annotations

import json

import pytest

import test_local_mcp_controller as tl
from brapi_client import TransportResponse
from part2_support import load_fixture
from test_local_mcp_controller import QUESTION, _part2_flow, saved, turn  # noqa: F401  (_part2_flow is autouse there)


class TwoPageHTTP(tl.SyntheticHTTP):
    """/observations and /observationunits: the six synthetic rows come as 3 + 3 over two pages; everything else as before."""

    def __call__(self, request):
        endpoint = request.url.removeprefix(tl.BASE)
        if endpoint not in ("/observations", "/observationunits"):
            return super().__call__(request)
        page = request.params.get("page", 0)
        assert page in (0, 1), f"a third page was requested: {page}"
        self.calls.append((endpoint, dict(request.params), request.read_timeout))
        _, body = load_fixture(self.replies[endpoint])
        payload = json.loads(body)
        rows = payload["result"]["data"]
        payload["result"]["data"] = rows[:3] if page == 0 else rows[3:]
        payload["metadata"]["pagination"].update(currentPage=page, pageSize=3, totalCount=len(rows), totalPages=2)
        return TransportResponse(200, {"Content-Type": "application/json"}, json.dumps(payload).encode("utf-8"))


@pytest.fixture(autouse=True)
def _two_page_server(monkeypatch):
    monkeypatch.setattr(tl, "SyntheticHTTP", TwoPageHTTP)


from test_local_mcp_controller import chat  # noqa: E402,F401  (after the server patch is registered)


def test_the_default_setting_reads_both_pages_and_the_answer_completes(chat):
    chat.warm()
    history, state = turn(chat, QUESTION)
    job = state["job"]
    assert job.controller.config.max_observation_pages == 2          # the app's own setting (the run setting's default)
    history, state = turn(chat, "approve", history, state)
    assert "Reply **accept**" in history[-1]["content"], job.controller.notes
    calls = [(c[0], c[1].get("page")) for c in chat.server.calls]
    assert calls == [("/observations", 0), ("/observations", 1), ("/observationunits", 0), ("/observationunits", 1)], calls
    turn(chat, "accept", history, state)
    result, _manifest = saved(job)
    assert result.execution_status == "completed", result.notes
    assert result.retrieval.complete and all(a.complete for a in result.retrieval.artifacts)
    assert sorted(a.row_count for a in result.retrieval.artifacts) == [6, 6]
    assert result.counts.live_attempts == 4
