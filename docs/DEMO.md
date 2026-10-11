# Running the demo

Everything runs on one machine. Nothing but the analysis call leaves it: the
redacted transcript goes to the configured LLM provider (Gemini by default).
Raw audio, the unredacted transcript and the database never do.

## What runs where

| Piece | Where | Port |
| --- | --- | --- |
| Postgres, Redis | Docker | 5433, 6379 |
| API (`migrate`, `api`), analysis worker, reaper (`beat`) | Docker, `service` profile | 8000 |
| Langfuse (tracing, no PII) | Docker | 3000 |
| ASR worker (mlx-whisper + pyannote) | **native**: Metal does not reach into Docker | — |
| Dashboard (Vite dev server, proxies `/api` to the API) | native | 5173 |

## Before the first run

- Docker Desktop, `uv`, Node 20+, `ffmpeg`.
- `uv sync --all-extras --dev` (or `make install`).
- `.env` at the repo root with at least `SAWTI_API_KEY`, `SAWTI_LLM_PROVIDER`
  (`gemini`), `GEMINI_API_KEY`, `HF_TOKEN` (with the pyannote diarization
  model's terms accepted on Hugging Face), and the `LANGFUSE_*` values.
- The first upload downloads the Whisper and pyannote weights (a few GB),
  so do one throwaway upload before recording.
- The Results page reads `data/serving_benchmark/comparison.json` and
  `data/memory_comparison/report.json`; without them it shows an error (503).

## Run

```bash
make demo
```

This brings the stack up (`make stack`, waits until healthy), then runs the
ASR worker and the dashboard in the foreground. The dashboard is given the API
key from `.env`; the reviewer id defaults to `qa-demo`
(`make demo DEMO_REVIEWER=qa-emad` to change it).

- Dashboard: <http://localhost:5173>
- API docs: <http://localhost:8000/docs>
- Langfuse: <http://localhost:3000>

Ctrl-C stops the ASR worker and the dashboard. The stack keeps running;
`make down` stops it.

### When something is off

- **Port 5173 in use**: a dashboard dev server is already running (`npm run dev`
  in another terminal). Stop it or use that one.
- **Upload stays `transcribing`**: the ASR worker is not running, or is still
  downloading weights. Its log is in the `make demo` terminal.
- **Upload `failed` with a diarization error**: `HF_TOKEN` missing, or the
  pyannote model's terms not accepted.
- **Upload `failed` with a 429**: the Gemini free-tier quota. Wait, or switch
  `SAWTI_LLM_PROVIDER`.

## Recording script (about 4 minutes)

Use the three recordings listed in `eval_results.md` (2026-10-11, demo-call
selection), or any from `data/audio/`. All calls are synthetic. Do one
throwaway upload first so no model download happens on camera.

1. **Results** (30 s). "Every number is per language, with its n and hardware."
   Point at the memory panel: the effect against Phase 4 ± noise, one
   pre-registered verdict per language. Switch the serving table between `ar`,
   `en`, `mixed`. Mention that accuracy is against Gemini-generated labels, so it
   is not a fair cross-provider comparison.
2. **New call** (60 s). Drop the first recording, pick its language (explain why
   there is no auto-detect: it garbles code-switched Arabic), upload. Follow the
   pipeline: transcription and diarization on this machine, PII redacted (the
   count shows), then analysis. Raw audio is deleted once transcription ends.
3. **Routed to review** (20 s). When coverage falls below 0.70 the pipeline says
   so; click **Open review**. If a call completes instead, say that this is the
   other branch (above threshold, no human needed) and upload the next one.
4. **Review** (90 s). The transcript is redacted (`[PHONE]`, `[CARD]`), each turn
   in its own direction. Click a claim to highlight its verbatim evidence. Show
   the "Rejected by grounding" panel: claims the model made whose quote is not in
   the transcript, dropped before anyone sees them as findings. Confirm most
   claims, **Correct** one with verbatim evidence ("✓ Found verbatim in turn n"
   appears once the quote matches), submit. The paused run resumes.
5. **Memory** (40 s). The correction becomes a candidate rule. If it contradicts
   an existing one it is held here; walk through the four resolutions.
6. **Arabic** (20 s). Toggle عربي: the whole UI flips to RTL. Transcript turns
   keep their own direction.

Things not to claim on camera: that grounding proves a claim is *right* (it
proves the quote exists; see the 2026-10-11 finding in `eval_results.md`), or
that the audio is real telephony (it is synthetic TTS).
