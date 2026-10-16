# Demo video

A ~90 s screen recording of the dashboard, produced by a script instead of by hand:
a visible cursor, smooth zooms on the parts that matter, and captions.

## Data: replayed from a real run

The browser's `/api` calls are answered from `data/demo_capture/{before,after}.json`,
saved from the live API around one real review (`scripts/capture_demo_api.py`).
The transcript, claims, coverage, PII count, memory rules and results on screen are
what Sawti produced. Only the pipeline's in-between statuses are rebuilt, from the
real call with its results hidden until it reaches them, and their timing is compressed.

If only `after.json` exists (the review was submitted before `before` was captured),
the pre-review call is the same analysis with the review removed and the status set
back to `awaiting_review`. Reviewing does not change the transcript, claims or coverage.

Without a capture it falls back to the e2e mock, and the title card says
"illustrative data". `DEMO_REQUIRE_CAPTURE=1` makes a missing capture an error.

## Record

```bash
cd frontend
npx playwright test -c demo_video/demo.config.ts   # frames → demo_video/out/frames
python3 demo_video/render.py                       # → demo_video/out/sawti-demo.mp4
```

Frames come from Chrome's screencast at 1920×1080. Zoom is a CSS transform on
`#root`, so text is re-rendered sharp rather than upscaled. Encoded H.264 CRF 17, 30 fps.

Env: `DEMO_CAPTURE` (capture folder), `DEMO_REPO_URL` (end card),
`DEMO_CHROMIUM` (browser executable, if Playwright's default is missing).

## Files

- `demo.spec.ts`: the storyboard (scenes, captions, timing).
- `overlay.ts`: cursor, click ripple, zoom, captions, title cards; injected outside `#root`.
- `source.ts`: replay of the captured responses, or the mock.
- `render.py`: frames to MP4 (ffmpeg).
