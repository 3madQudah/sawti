# 12-TESTING

How this project is tested, and what the test suite is and is not evidence of.

## Running

```bash
make test        # pytest across tests/
make lint        # ruff check .
make typecheck   # mypy src
```

`pytest` is configured in `pyproject.toml` with `asyncio_mode = "auto"`, so
`async def test_*` functions run without a decorator. Tests live under `tests/`
mirroring `src/sawti/` one file per module — `src/sawti/agent/nodes/ground.py`
is tested by `tests/agent/nodes/test_ground.py`, and nothing else.

## The rule

**No code without a corresponding pytest test.** A module with no test file is
not finished. A scaffolded stub that still raises `NotImplementedError` carries
a test file of skipped placeholder tests, each with a docstring stating what the
real test will assert — so the shape of the missing work is visible in
`pytest -ra` output rather than only in someone's head.

Skips are meaningful here. `pytest -ra` currently reports skips tagged with the
phase that will implement them (`phase 3`, `phase 4`), plus one tagged
`needs NER` — see below.

## What every stage ships

Each implemented module ships tests covering, at minimum:

1. **The happy path** — the thing works on well-formed input.
2. **At least one failure path** — the interesting half. A claim with no
   matching quote. A confidence score sitting exactly on the threshold. A
   provider that raises. A transcript file that is missing. State missing the
   key a router reads.
3. **The invariant it exists to protect**, asserted directly rather than
   implied.

Boundary cases are pinned explicitly where a future tidy-up could silently flip
them. `test_confidence_exactly_on_the_threshold_auto_passes` exists because
`<` versus `<=` is a one-character change that alters who sees a human.

## What the tests deliberately do not do

- **They never call a live model.** Every test that would reach an LLM patches
  `get_llm_provider` with a stub that records its arguments and returns a fixed
  payload. Tests assert what the node *sent* and what it *did with the reply* —
  never model quality. Model quality is measured by evaluation, not tests.
- **They do not need a database or Redis.** The graph's checkpointer defaults to
  an in-process `MemorySaver`, so human-in-the-loop interrupts and resumes are
  tested end to end with no services running.
- **They do not assert on wall-clock timing or network behaviour.**

## Testing the graph

Graph tests run against the compiled graph rather than calling node functions
directly, because the things worth testing — routing, suspension, resume — only
exist once nodes are wired.

- **Which branch a run took** is read from `astream(..., stream_mode="updates")`,
  where each chunk is keyed by the node that produced it. A suspended run's final
  chunk is `__interrupt__`, not a node name.
- **Every run needs a `thread_id`.** The graph is checkpointed, so tests pass a
  fresh `configurable.thread_id` per run — sharing one would let runs read each
  other's state.
- **Interrupt payloads** hang off the paused task, not the invoke result:
  `(await graph.aget_state(config)).tasks[0].interrupts[0].value`. `ainvoke`
  returns the state as of the suspension.
- **Escalation is driven through the real mechanism**, never by seeding
  `requires_human_review`. A test that wants a human review makes the stub
  provider paraphrase a quote, so grounding drops it, coverage falls, and the
  conditional edge does the escalating. Seeding the flag would test nothing,
  since `assess_confidence` overwrites it.

## A test that is meant to stay red

`tests/privacy/test_redaction.py::test_redact_removes_names_in_arabic_and_english`
is skipped on purpose. `redact()` is a regex MVP and catches structured
identifiers only; personal names need NER. The test is kept, skipped, with a
reason pointing at `docs/09-DECISIONS.md`, so the gap appears in every test run
instead of being quietly deleted. Do not remove it to make the output tidy.

## Testing the audio pipeline

Neither Whisper, pyannote, nor the TTS endpoint runs in the test suite. All
three are stubbed at their own module boundary, so the tests stay fast and
offline while still covering the code that can actually be wrong.

- **`tests/asr/test_transcribe.py`** stubs both Whisper backends. It pins what
  this repo owns — that the language is *forced* rather than detected, that
  `AUTO_DETECT` is the only path that stops forcing, that segments come back
  ordered, that a missing file raises — not Whisper's accuracy.
- **`tests/asr/test_diarization.py`** stubs the gated pyannote pipeline. It
  covers ordering, the missing-token error path, greatest-overlap alignment,
  and the no-overlap fallback. It does **not** cover pyannote's clustering
  quality, which is unmeasured because the weights are gated — see
  `docs/09-DECISIONS.md`.
- **`tests/data/test_generate_audio.py`** stubs edge-tts and ffmpeg. Its
  load-bearing assertion is that the manifest's speaker timeline matches the
  duration of the WAV actually written: if those drift, every diarization
  number is silently scored against the wrong times. It also covers resume and
  TTS retry, because a corpus run is ~45 minutes against a public network
  service and has already died once mid-run.
- **`tests/data/test_transcript_parser.py`** runs over all 150 real
  transcripts, asserting every line is attributable. It also asserts each entry
  in `SPEAKER_REPAIRS` still targets a genuinely defective line, so a stale
  repair cannot silently relabel a good one.

Accuracy claims — WER, diarization accuracy, the ASR propagation delta — are
not unit tests. They come from `sawti.eval.experiments` runs against the real
corpus and live in `eval_results.md`.

## Known gaps

- `make lint` and `make test` are clean. `make typecheck` reports two
  pre-existing errors in `src/sawti/api/tasks.py` from Celery's missing type
  stubs — untouched service-phase scaffolding, not new.
- The service phase (API, database) and phase 4 (memory) modules are stubs with
  skipped placeholder tests.
- No end-to-end test runs the real provider. That is what
  `sawti.eval.experiments` is for, and its results live in `eval_results.md`.
- **`sawti.asr.diarization` has never been executed against real audio.** Its
  tests stub the pipeline, so they prove the module's contract and nothing
  about whether pyannote separates these speakers. Unblocking needs `HF_TOKEN`.
