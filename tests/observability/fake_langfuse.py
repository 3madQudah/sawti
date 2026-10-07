"""A stand-in Langfuse client that records every payload it is given (phase 6.3 tests)."""

from __future__ import annotations

import json
from typing import Any


class FakeObservation:
    def __init__(self, client: FakeLangfuse, kind: str, payload: dict[str, Any]) -> None:
        self._client = client
        self.kind = kind
        self.payload = payload
        self.ended: dict[str, Any] | None = None

    def span(self, **payload: Any) -> FakeObservation:
        return self._client._record("span", payload)

    def generation(self, **payload: Any) -> FakeObservation:
        return self._client._record("generation", payload)

    def end(self, **payload: Any) -> None:
        self.ended = payload
        self._client.sent.append(("end", payload))


class FakeLangfuse:
    def __init__(self) -> None:
        self.observations: list[FakeObservation] = []
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.flushed = 0

    def _record(self, kind: str, payload: dict[str, Any]) -> FakeObservation:
        observation = FakeObservation(self, kind, payload)
        self.observations.append(observation)
        self.sent.append((kind, payload))
        return observation

    def trace(self, **payload: Any) -> FakeObservation:
        return self._record("trace", payload)

    def generation(self, **payload: Any) -> FakeObservation:
        return self._record("generation", payload)

    def flush(self) -> None:
        self.flushed += 1

    def of(self, kind: str) -> list[FakeObservation]:
        return [o for o in self.observations if o.kind == kind]

    def everything_sent(self) -> str:
        """Every payload, serialized — what a PII assertion searches."""
        return json.dumps([[kind, payload] for kind, payload in self.sent], default=str, ensure_ascii=False)
