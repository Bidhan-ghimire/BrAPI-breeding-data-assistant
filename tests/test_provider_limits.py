"""Provider limits with synthetic SDK exceptions and scripted models; no network."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import anyio
import httpx
import openai
import pytest

import chat_presenter as presenter
import llm
from contracts import AnalysisReport, ToolResult


FINAL = '{"status":"completed","payload":{"answer":5}}'


def sdk_error(body, *, headers=None, status=429):
    response = httpx.Response(status, headers=headers or {}, request=httpx.Request("POST", "https://synthetic.invalid/v1/chat/completions"))
    return openai.APIStatusError("SYNTHETIC private raw body must not be copied", response=response, body=body)


def translated(body, *, headers=None, status=429):
    async def create(**kwargs):
        raise sdk_error(body, headers=headers, status=status)

    async def call():
        client = llm.OpenAICompatibleClient(llm.LlmSettings(model="synthetic-model"))
        client._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with pytest.raises(llm.ModelProviderError) as caught:
            await client.complete([], [], max_tokens=10)
        return caught.value

    return anyio.run(call)


class Clock:
    def __init__(self):
        self.value = 0.0
        self.sleeps = []

    def __call__(self):
        return self.value

    async def sleep(self, delay):
        self.sleeps.append(delay)
        self.value += delay


class Tools:
    def __init__(self):
        self.calls = []

    async def schemas(self):
        return [{"name": "add", "description": "SYNTHETIC addition", "inputSchema": {
            "type": "object", "additionalProperties": False, "required": ["a", "b"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}}]

    async def call(self, name, args):
        self.calls.append((name, dict(args)))
        return ToolResult(ok=True, complete=True, data={"sum": args["a"] + args["b"]})


def run(tmp_path, replies, *, budget=None, clock=None, tools=None, agent="synthetic", jitter=lambda: 0.0):
    model = llm.FakeModelClient(replies)
    timer = clock or Clock()
    selected = tools or Tools()

    async def go():
        return await llm.run_agent_loop(model, selected, agent_name=agent, system_prompt="SYNTHETIC", user_message="Add 2 and 3",
                                        budget=budget or llm.Budget(), log_dir=tmp_path, run_id="provider_test",
                                        model_requested="synthetic-model", clock=timer, sleep=timer.sleep, jitter=jitter)

    result = anyio.run(go)
    events = [json.loads(line) for line in (tmp_path / result.log_path).read_text(encoding="utf-8").splitlines()]
    return result, model, timer, selected, events


@pytest.mark.parametrize("body,expected,retryable", [
    ({"code": "insufficient_quota"}, "quota", False),
    ({"type": "billing_hard_limit_reached"}, "quota", False),
    ({"error": {"code": "insufficient_quota", "type": "rate_limit_exceeded"}}, "quota", False),
    ({"code": "rate_limit_exceeded", "type": "tokens"}, "rate_limit", True),
    ({"error": {"code": "rate_limit_exceeded"}}, "rate_limit", True),
    ({"type": "rate_limit_error"}, "rate_limit", True),
    ({"code": "slow_down"}, "rate_limit", True),
    ({"message": "Rate limit exceeded. SYNTHETIC raw secret"}, "rate_limit_unknown", False),
    ({"code": "future_unknown_code"}, "rate_limit_unknown", False),
])
def test_adapter_classifies_only_structured_allowlisted_429_codes(body, expected, retryable):
    error = translated(body, headers={"retry-after": "3"})
    assert error.code == expected and error.retryable is retryable and error.status_code == 429
    assert "SYNTHETIC" not in str(error) and "future_unknown_code" not in str(error)
    if expected == "rate_limit":
        assert error.retry_after_seconds == 3.0


def test_provider_opt_out_is_respected_and_non429_stays_generic():
    error = translated({"code": "rate_limit_exceeded"}, headers={"x-should-retry": "false"})
    assert error.code == "rate_limit" and not error.retryable
    other = translated({"code": "rate_limit_exceeded"}, status=403)
    assert other.code == "http_error" and not other.retryable and "HTTP 403" in other.message


@pytest.mark.parametrize("headers,expected", [
    ({"retry-after": "7.5"}, 7.5),
    ({"retry-after-ms": "1250", "retry-after": "9"}, 9.0),
    ({"retry-after-ms": "12500", "retry-after": "9"}, 12.5),
    ({"retry-after-ms": "30000", "retry-after": "Wed, 30 Sep 2026 03:00:20 GMT"}, 30.0),
    ({"retry-after": "Wed, 30 Sep 2026 03:00:20 GMT"}, 20.0),
    ({"retry-after": "Wed, 30 Sep 2026 02:59:00 GMT"}, 0.0),
    ({"retry-after": "120"}, 120.0),
    ({"retry-after": "NaN"}, None),
    ({"retry-after": "Infinity"}, None),
    ({"retry-after": "-2"}, None),
    ({"retry-after": "SYNTHETIC not a date"}, None),
])
def test_retry_after_is_finite_exact_and_never_shortens_long_waits(headers, expected):
    now = datetime(2026, 9, 30, 3, 0, 0, tzinfo=timezone.utc)
    assert llm._retry_after_seconds(headers, now=now) == expected


@pytest.mark.parametrize("body", [{"code": "insufficient_quota"}, {"message": "generic HTTP429"}])
def test_quota_and_unknown_429_never_retry_or_claim_temporary_cause(tmp_path, body):
    error = translated(body, headers={"retry-after": "1"})
    result, model, timer, _, events = run(tmp_path, [error, llm.reply_text(FINAL)])
    assert result.status == "failed" and len(model.requests) == 1 and timer.sleeps == []
    assert not any(e["event"] == "provider_retry" for e in events)
    message = result.errors[0].message
    assert "temporary rate limit" not in message
    assert "no automatic retry" in message.lower()
    shown = presenter.answer_summary(SimpleNamespace(execution_status="failed", retrieval=None, notes=[],
        analysis=AnalysisReport(status="failed", caveats=["agent error " + result.errors[0].code + ": " + message])))
    assert "Could not complete this question" in shown and "HTTP 429" in shown
    if error.code == "rate_limit_unknown":
        assert "cause is unknown" in shown
    else:
        assert "quota" in shown and "billing" in shown


def test_retry_preserves_transcript_and_never_repeats_executed_tools(tmp_path):
    error = translated({"code": "rate_limit_exceeded"}, headers={"retry-after": "7.5"})
    budget = llm.Budget()
    result, model, timer, tools, events = run(tmp_path, [
        llm.reply_tools([("add", '{"a":2,"b":3}')]), error, llm.reply_text(FINAL)], budget=budget)
    assert result.status == "completed" and result.payload == {"answer": 5}
    assert timer.sleeps == [7.5] and budget.provider_retries == 1 and result.model_calls == 3
    assert tools.calls == [("add", {"a": 2, "b": 3})] and result.tool_calls == 1
    assert model.requests[1]["messages"] == model.requests[2]["messages"]
    assert result.usage is None  # the failed attempt's use was not reported
    assert [e["retry_number"] for e in events if e["event"] == "provider_retry"] == [1]
    assert "SYNTHETIC private raw body" not in json.dumps(events)


def test_two_retry_allowance_is_shared_across_agents(tmp_path):
    error = translated({"code": "rate_limit_exceeded"})
    budget, timer = llm.Budget(), Clock()
    first, _, _, _, _ = run(tmp_path, [error, error, llm.reply_text(FINAL)], budget=budget, clock=timer, agent="first")
    assert first.status == "completed" and timer.sleeps == [2.0, 4.0]
    second, model, _, _, _ = run(tmp_path, [error, llm.reply_text(FINAL)], budget=budget, clock=timer, agent="second")
    assert second.status == "failed" and len(model.requests) == 1
    assert budget.model_calls == 4 and budget.provider_retries == 2 and timer.sleeps == [2.0, 4.0]
    assert "shared automatic-retry limit" in second.errors[0].message


def test_fallback_backoff_has_bounded_injectable_jitter(tmp_path):
    error = translated({"code": "rate_limit_exceeded"})
    result, _, timer, _, _ = run(tmp_path, [error, error, llm.reply_text(FINAL)], jitter=lambda: 0.75)
    assert result.status == "completed" and timer.sleeps == [2.75, 4.75]


@pytest.mark.parametrize("noise,expected", [(-1.0, 2.0), (500.0, 3.0), (float("nan"), 2.0), (float("inf"), 2.0)])
def test_jitter_cannot_make_wait_negative_nonfinite_or_unbounded(tmp_path, noise, expected):
    error = translated({"code": "rate_limit_exceeded"})
    result, _, timer, _, _ = run(tmp_path, [error, llm.reply_text(FINAL)], jitter=lambda: noise)
    assert result.status == "completed" and timer.sleeps == [expected]


def test_valid_retry_after_is_not_reduced_or_replaced_by_backoff_jitter(tmp_path):
    error = translated({"code": "rate_limit_exceeded"}, headers={"retry-after-ms": "1000", "retry-after": "8"})
    result, _, timer, _, _ = run(tmp_path, [error, llm.reply_text(FINAL)], jitter=lambda: 0.99)
    assert result.status == "completed" and timer.sleeps == [8.0]


@pytest.mark.parametrize("budget", [llm.Budget(max_model_calls=1), llm.Budget(max_elapsed_seconds=1.0)])
def test_retry_cannot_exceed_shared_model_or_time_budget(tmp_path, budget):
    error = translated({"code": "rate_limit_exceeded"}, headers={"retry-after": "2"})
    result, model, timer, _, _ = run(tmp_path, [error, llm.reply_text(FINAL)], budget=budget)
    assert result.status == "limit_reached" and len(model.requests) == 1 and timer.sleeps == []
    assert budget.provider_retries == 0 and any(e.code == "budget_exhausted" for e in result.errors)


def test_long_provider_wait_stops_without_shortening_it(tmp_path):
    error = translated({"code": "rate_limit_exceeded"}, headers={"retry-after": "61"})
    result, model, timer, _, _ = run(tmp_path, [error, llm.reply_text(FINAL)])
    assert result.status == "failed" and len(model.requests) == 1 and timer.sleeps == []
    assert "wait was not shortened" in result.errors[0].message


def test_retry_sleep_rechecks_elapsed_budget_before_next_call(tmp_path):
    class Overslept(Clock):
        async def sleep(self, delay):
            await super().sleep(delay)
            self.value += 1000

    error = translated({"code": "rate_limit_exceeded"})
    result, model, timer, _, _ = run(tmp_path, [error, llm.reply_text(FINAL)], clock=Overslept())
    assert result.status == "limit_reached" and len(model.requests) == 1 and timer.sleeps == [2.0]


def test_cancellation_during_retry_wait_sends_no_next_request(tmp_path):
    error = translated({"code": "rate_limit_exceeded"})
    model = llm.FakeModelClient([error, llm.reply_text(FINAL)])

    async def go():
        with anyio.CancelScope() as scope:
            async def cancel(delay):
                scope.cancel()
                await anyio.sleep(0)

            await llm.run_agent_loop(model, Tools(), agent_name="cancel", system_prompt="SYNTHETIC", user_message="SYNTHETIC",
                                      budget=llm.Budget(), log_dir=tmp_path, run_id="provider_test", model_requested="synthetic-model", sleep=cancel)
            pytest.fail("Cancellation must propagate from the retry wait")

    anyio.run(go)
    assert len(model.requests) == 1
    events = (tmp_path / "provider_test" / "llm_cancel.jsonl").read_text(encoding="utf-8")
    assert '"event": "cancelled"' in events


def test_model_await_is_bounded_by_remaining_work_time(tmp_path):
    async def go():
        model = llm.FakeModelClient([llm.reply_text(FINAL)], gate=anyio.Event())
        return await llm.run_agent_loop(model, Tools(), agent_name="timeout", system_prompt="SYNTHETIC", user_message="SYNTHETIC",
                                        budget=llm.Budget(max_elapsed_seconds=0.02), log_dir=tmp_path,
                                        run_id="provider_test", model_requested="synthetic-model")

    result = anyio.run(go)
    assert result.status == "limit_reached" and result.model_calls == 1
    assert result.errors[0].code == "budget_exhausted" and "expired while waiting" in result.errors[0].message


def test_sdk_auto_retries_remain_disabled(monkeypatch):
    captured = []

    class SDK:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        async def close(self):
            pass

    monkeypatch.setattr(openai, "AsyncOpenAI", SDK)

    async def go():
        async with llm.OpenAICompatibleClient(llm.LlmSettings(model="synthetic-model")):
            pass

    anyio.run(go)
    assert captured[0]["max_retries"] == 0
