"""Tests for `sawti.llm.usage` (phase 6.3)."""

from __future__ import annotations

import asyncio

from sawti.llm.usage import Usage, capture_usage, record_usage, total


def test_record_outside_a_capture_is_a_no_op() -> None:
    record_usage(1, 2)  # must not raise or leak anywhere


def test_capture_collects_every_request_and_total_sums_them() -> None:
    with capture_usage() as usage:
        record_usage(10, 3, model="m1")
        record_usage(12, None, model="m2")  # a retry that did not report output tokens
    assert total(usage) == Usage(prompt_tokens=22, completion_tokens=3, model="m2")
    assert total([]) == Usage()


async def test_concurrent_captures_do_not_see_each_other() -> None:
    """The benchmark runs 16 calls at once in one event loop; each must count only its own tokens."""

    async def one(n: int) -> int | None:
        with capture_usage() as usage:
            await asyncio.sleep(0.01 * (5 - n))
            record_usage(n, n)
            await asyncio.sleep(0)
        return total(usage).prompt_tokens

    assert await asyncio.gather(*(one(n) for n in range(5))) == [0, 1, 2, 3, 4]


def test_a_nested_capture_does_not_swallow_usage_from_the_outer_one() -> None:
    """The tracer captures inside the benchmark's capture; both must see the request."""
    with capture_usage() as outer:
        with capture_usage() as inner:
            record_usage(7, 2)
        record_usage(1, 1)
    assert [u.prompt_tokens for u in inner] == [7]
    assert [u.prompt_tokens for u in outer] == [7, 1]
