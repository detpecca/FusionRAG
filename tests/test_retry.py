"""限流重试测试: 429 指数退避重试、重试耗尽后抛错、非可重试错误直接抛。"""

import pytest

from fusionrag.llm import LLMService, _is_retryable


class FakeRateLimitError(Exception):
    status_code = 429


class FakeServerError(Exception):
    status_code = 500


class FakeBadRequest(Exception):
    status_code = 400


def test_retryable_classification():
    assert _is_retryable(FakeRateLimitError("限流"))
    assert _is_retryable(FakeServerError("服务端错误"))
    assert not _is_retryable(FakeBadRequest("参数错误"))
    assert not _is_retryable(ValueError("普通异常"))


async def test_chat_retry_recovers(test_config):
    test_config.retry_base_delay = 0.01
    calls = {"n": 0}

    async def flaky(prompt, **kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeRateLimitError("429")
        return "ok"

    test_config.llm_func = flaky
    llm = LLMService(test_config)
    assert await llm.chat("hi") == "ok"
    assert calls["n"] == 3


async def test_chat_retry_exhausted(test_config):
    test_config.retry_base_delay = 0.01
    test_config.llm_max_retries = 2
    calls = {"n": 0}

    async def always_fail(prompt, **kwargs):
        calls["n"] += 1
        raise FakeRateLimitError("429")

    test_config.llm_func = always_fail
    llm = LLMService(test_config)
    with pytest.raises(FakeRateLimitError):
        await llm.chat("hi")
    assert calls["n"] == 3  # 1 次原始 + 2 次重试


async def test_chat_no_retry_on_bad_request(test_config):
    test_config.retry_base_delay = 0.01
    calls = {"n": 0}

    async def bad_request(prompt, **kwargs):
        calls["n"] += 1
        raise FakeBadRequest("400")

    test_config.llm_func = bad_request
    llm = LLMService(test_config)
    with pytest.raises(FakeBadRequest):
        await llm.chat("hi")
    assert calls["n"] == 1  # 不可重试错误直接抛出


async def test_embed_retry_recovers(test_config):
    test_config.retry_base_delay = 0.01
    calls = {"n": 0}

    async def flaky_embed(texts):
        calls["n"] += 1
        if calls["n"] < 2:
            raise FakeServerError("500")
        return [[1.0, 2.0]] * len(texts)

    test_config.embedding_func = flaky_embed
    llm = LLMService(test_config)
    vectors = await llm.embed(["a", "b"])
    assert vectors.shape == (2, 2)
    assert calls["n"] == 2
