.PHONY: install test lint typecheck up down fmt audio transcribe asr-eval asr-forcing

install:
	uv sync --all-extras --dev

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

typecheck:
	uv run mypy src

up:
	docker compose up -d

down:
	docker compose down

# --- Phase 3: audio pipeline ---------------------------------------------
# Order matters: audio -> transcribe -> asr-eval. Each writes what the next
# one reads. `audio` and `transcribe` are long (~45 min and ~2 h on an M1);
# both resume, so a re-run after a failure picks up where it stopped.

# Synthesize call audio from the Phase 1 transcripts, with the speaker-timeline
# manifest the WER and diarization checks are scored against. Needs ffmpeg.
audio:
	uv run python -m sawti.data.generate_audio

# Transcribe the corpus with the language forced, and write per-language WER.
# Produces data/asr_transcripts/ for the propagation eval.
transcribe:
	uv run python -m sawti.eval.experiments.asr_propagation --transcribe-only

# Rerun the Phase 2 agent over the ASR transcripts and append the scores to
# eval_results.md. This one costs LLM calls.
asr-eval:
	uv run python -m sawti.eval.experiments.asr_propagation

# Forced-Arabic vs Whisper auto-detect, on the code-switched calls.
asr-forcing:
	uv run python -m sawti.eval.experiments.asr_language_forcing --categories mixed --limit 15
