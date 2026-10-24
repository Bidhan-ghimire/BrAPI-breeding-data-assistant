"""Prompt caching on the Anthropic path (Bidhan's decision, coach 2026-10-24 23:00 UTC, item (29)(b) step 1). Every request marks
the system prompt, the last tool definition and the last block of the newest turn with cache_control {"type": "ephemeral"} (the
5-minute cache), so each call reads from cache what the calls before it wrote: the API renders tools, then system, then messages,
and a marker caches everything before it. Three markers of the four allowed. The request is otherwise the one sent before; the
OpenAI path sends no marker. The 24 Oct Opus measurement paid the full input price on every call (cache_write 0, cache_read 0):
A36.4's 250,464 input tokens were $1.00 of its $1.09. Also: Sonnet 5.5's cache reads cost $0.20 per million tokens (the bundled
Claude API reference, shared/models.md), not the $0.10 the price table had. Everything here is SYNTHETIC: a fake SDK or a mock
transport, no network, no model."""
import json

import anthropic
import anyio
import httpx2
import pytest

import llm
import triage
from llm import LlmSettings
from test_anthropic_provider import (SYSTEM, USER, asked_tools, claude_settings, markers, message, message_json,
                                     no_sdk_variables_from_the_shell, open_client, result_for, run, stripped, text, tool_use)  # noqa: F401
from test_eval_harness import harness

MARK = {"type": "ephemeral"}
TOOLS = [{"type": "function", "function": {"name": name, "description": f"SYNTHETIC {name}",
                                           "parameters": {"type": "object", "properties": {}}}} for name in ("first", "second", "third")]


def test_the_system_prompt_the_last_tool_and_the_newest_turn_are_marked():
    request = llm.to_anthropic_request([SYSTEM, USER], TOOLS)
    assert request["system"] == [{"type": "text", "text": "S", "cache_control": MARK}]
    assert [tool.get("cache_control") for tool in request["tools"]] == [None, None, MARK]
    assert request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "q", "cache_control": MARK}]}]
    assert len(markers(request)) == 3                      # of the four the API allows


def test_tool_results_carry_the_marker_on_their_last_block_and_earlier_turns_none():
    request = llm.to_anthropic_request([SYSTEM, USER, asked_tools("t1", "t2"), result_for("t1"), result_for("t2")], TOOLS)
    last = request["messages"][-1]["content"]
    assert [block.get("cache_control") for block in last] == [None, MARK]
    assert markers({"earlier": request["messages"][:-1]}) == []
    assert stripped(request)["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "t1", "content": '{"ok": true}'},
        {"type": "tool_result", "tool_use_id": "t2", "content": '{"ok": true}'}]}


def test_the_request_is_otherwise_the_one_sent_before():
    """Without the markers: the system prompt as its text, the user turn as its text, the tools as converted before (the same
    schema objects), and no new top-level field."""
    request = llm.to_anthropic_request([SYSTEM, USER], TOOLS)
    plain = stripped(request)
    assert set(request) == {"system", "messages", "tools"}
    assert plain == {"system": "S", "messages": [{"role": "user", "content": "q"}],
                     "tools": [{"name": t["function"]["name"], "description": t["function"]["description"],
                                "input_schema": t["function"]["parameters"]} for t in TOOLS]}
    assert all(sent["input_schema"] is tool["function"]["parameters"] for sent, tool in zip(request["tools"], TOOLS))


def test_an_empty_system_prompt_is_sent_as_before_without_a_marker():
    """The API refuses an empty text block, so an empty system prompt stays the empty string it was."""
    request = llm.to_anthropic_request([{"role": "system", "content": ""}, USER], [])
    assert request["system"] == "" and dict(markers(request)) == {"request.messages[0].content[0]": MARK}


def test_each_request_of_a_loop_marks_its_newest_turn_and_keeps_the_rest_append_only(tmp_path):
    first = message([text("Adding."), tool_use("toolu_a", 2, 3)], stop="tool_use")
    client, sdk = open_client(claude_settings(tmp_path), [first, message([text('{"status": "completed", "payload": {"answer": 5}}')])])
    result, _ = run(client, tmp_path)
    assert result.status == "completed" and len(sdk.sent) == 2
    for sent in sdk.sent:
        where = [w for w, _ in markers(sent) if w.startswith("request.messages")]
        assert len(markers(sent)) == 3 and where == [f"request.messages[{len(sent['messages']) - 1}].content[0]"]
    before, after = (stripped(sent)["messages"] for sent in sdk.sent)
    assert after[:len(before)] == before                                          # append-only, the markers aside
    assert sdk.sent[0]["system"] == sdk.sent[1]["system"] and sdk.sent[0]["tools"] == sdk.sent[1]["tools"]   # the cached prefix


def test_a_long_loop_keeps_three_markers_on_every_request(tmp_path):
    """Review 2026-10-24, SHOULD-FIX 1: earlier turns that kept their marker would send 4, 5, ... markers, and the API allows 4;
    the two-request loop above could not tell. Four tool turns, then the final answer: five requests."""
    calls = [message([text("Adding."), tool_use(f"toolu_{i}", i, 1)], stop="tool_use") for i in range(4)]
    client, sdk = open_client(claude_settings(tmp_path), calls + [message([text('{"status": "completed", "payload": {"answer": 5}}')])])
    result, _ = run(client, tmp_path)
    assert result.status == "completed" and len(sdk.sent) == 5
    for sent in sdk.sent:
        assert len(markers(sent)) == 3
        assert [w for w, _ in markers(sent) if w.startswith("request.messages")] == [
            f"request.messages[{len(sent['messages']) - 1}].content[{len(sent['messages'][-1]['content']) - 1}]"]


def test_texts_are_sent_as_they_are():
    """A system prompt keeps its own whitespace; no system field is added without a system message; an empty user turn stays
    the empty string it was, without a marker (the API refuses an empty text block)."""
    assert llm.to_anthropic_request([{"role": "system", "content": "S\n"}, USER], [])["system"][0]["text"] == "S\n"
    assert "system" not in llm.to_anthropic_request([USER], [])
    empty = llm.to_anthropic_request([SYSTEM, {"role": "user", "content": ""}], [])
    assert empty["messages"] == [{"role": "user", "content": ""}] and dict(markers(empty)) == {"request.system[0]": MARK}


def test_the_real_sdk_sends_the_markers_in_the_request_body(tmp_path):
    """The exact bytes the SDK sends (a mock transport, no socket)."""
    seen: list[httpx2.Request] = []

    def answer(request):
        seen.append(request)
        return httpx2.Response(200, json=message_json("Hi"))

    http = anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(answer))
    model = llm.AnthropicClient(claude_settings(tmp_path), http_client=http)
    anyio.run(triage._complete, model, "PROMPT", "hello")
    body = json.loads(seen[0].content)
    assert body["system"] == [{"type": "text", "text": "PROMPT", "cache_control": MARK}]
    assert body["messages"] == [{"role": "user", "content": [{"type": "text", "text": "hello", "cache_control": MARK}]}]


def test_the_openai_path_sends_no_cache_marker():
    sent: list[dict] = []

    class Msg:
        content, tool_calls, refusal = "done", [], None

    class Choice:
        message, finish_reason = Msg(), "stop"

    class Resp:
        choices, model, usage = [Choice()], "m", None

    class Completions:
        async def create(self, **kwargs):
            sent.append(kwargs)
            return Resp()

    class Sdk:
        chat = type("Chat", (), {"completions": Completions()})()

    async def send():
        client = llm.OpenAICompatibleClient(LlmSettings(model="m", base_url="https://api.openai.com/v1", allow_remote=True))
        client._client = Sdk()
        return await client.complete([{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}], TOOLS, max_tokens=5)

    anyio.run(send)
    assert "cache_control" not in json.dumps(sent[0]) and sent[0]["messages"][0] == {"role": "system", "content": "S"}


def test_sonnet_5_5_cache_reads_are_priced_as_the_claude_api_reference_gives_them():
    """shared/models.md: Sonnet 5.5 "$2 / $10 per MTok; cache reads $0.20"; Opus 5.5 "$4 / $20 per MTok; cache reads $0.20".
    A cache write is 1.25 x input (the 5-minute cache the app marks)."""
    assert harness.PRICES["claude-sonnet-5-5"] == {"input": 2.00, "cache_write": 2.50, "cache_read": 0.20, "output": 10.00}
    assert harness.PRICES["claude-opus-5-5"] == {"input": 4.00, "cache_write": 5.00, "cache_read": 0.20, "output": 20.00}
    assert harness.PRICES["claude-haiku-5-5"] == {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50}  # review
    assert harness.PRICES["gpt-6-luna"] == {"input": 0.10, "cache_write": 0.125, "cache_read": 0.01, "output": 0.50}
    reply = harness.reply_tokens("claude-sonnet-5-5", None, {"input_tokens": 1000, "cache_creation_input_tokens": 2000,
                                                             "cache_read_input_tokens": 10000, "output_tokens": 100})
    assert harness.tally([reply], calls=1)["cost_usd"] == pytest.approx((1000 * 2.00 + 2000 * 2.50 + 10000 * 0.20 + 100 * 10.00) / 1e6)
