# 16 — Style guide

These are conventions the code already follows. When in doubt, match the
surrounding file.

## Python

| Tool | Config (`pyproject.toml`) |
|---|---|
| ruff lint | line length 110, target py311, rules `E F I UP B N ANN RUF` |
| ruff ignores | `ANN401`, `ANN101/102` (deprecated upstream), `RUF001/2/3`, which fire on almost every Arabic literal. `ANN` is off in `tests/`. |
| mypy | `strict = true`, pydantic plugin, `mypy_path = src`. `scripts/` and `notebooks/` are excluded. Untyped third-party packages (whisper, pyannote, vllm, langfuse, celery…) are `ignore_missing_imports`. |

- **Every value that crosses a node or API boundary is a Pydantic v2
  model.** No raw dicts. Invariants go in validators, so an invalid object
  can't exist (`Claim.evidence`, `Quote` offsets, `CallAnalysis` routing).
- **Configuration** comes only from `sawti.config.get_settings()`. Never read
  `os.environ` directly.
- **Per-language results.** Metric functions return `dict[Language, float]`,
  so a blended number can't slip through.
- **Docstrings** state the role, the arguments, what is returned and what is
  raised. Module docstrings say which phase the module belongs to and what it
  deliberately does not do.
- **Comments** explain *why*, especially when an ordering or a default is
  load-bearing (see `sawti.agent.graph`). They don't narrate code.
- **Names** that two places must agree on are constants (node names, task
  names, queue names).
- **Heavy imports** (torch, the graph, sentence-transformers) stay inside the
  functions that need them wherever the API image must not carry them.
- **PII.** Any type holding raw text hides it from `repr`
  (`AsrResult.raw_transcript`). Never log, trace or store raw transcript text.

## Frontend (`frontend/`)

- React 18, TypeScript, Vite, Tailwind, TanStack Query. The API client is
  generated from `openapi.json` (`openapi-typescript` + `openapi-fetch`), and
  `npm run typecheck` fails when it is stale.
- **i18n.** All UI text lives in `src/i18n/strings.ts` as an `en` and an `ar`
  dictionary with identical keys. A test checks that every key exists in both,
  that none is empty, and that the placeholders match. The backend sends codes
  and numbers, never UI prose, and the frontend translates them.
- **Direction.**
  - The UI language sets `<html dir lang>`: English is LTR, Arabic is RTL.
  - Each transcript turn sets its own direction from its per-turn language
    (`lib/direction.ts`: `ar` → `rtl`, `en` → `ltr`, `mixed` → `auto`),
    independent of the UI language.
  - Free text with no language tag (rule text, conflict reasoning) takes the
    direction of the script with more letters, not `dir="auto"`.
  - Ids, numbers and code are `dir="ltr"` inside RTL layouts.
  - Use logical Tailwind classes only (`ms-`, `me-`, `ps-`, `pe-`,
    `text-start`). The source has no `ml-`/`mr-`/`pl-`/`pr-`/`text-left`/`text-right`.
- **Tests.** Vitest and Testing Library for components. Playwright runs each
  page in EN and AR against an in-memory mock API (`e2e/mockApi.ts`), plus a
  1024 px check that the page doesn't scroll horizontally.
- **Fonts.** IBM Plex Sans, IBM Plex Sans Arabic, IBM Plex Mono, self-hosted
  via `@fontsource`.
- The page designs are in [`design/`](design/).

## Docs

- Numbered files in `docs/`. `eval_results.md` is the record of numbers and
  [09-DECISIONS.md](09-DECISIONS.md) is the record of reasoning. The other docs
  summarize and link to them.
- **Every number has a source link** to an `eval_results.md` round or a
  `data/` file, and is reported per language. Never a blended figure.
- **State limits next to results**: n, hardware, synthetic data,
  LLM-generated labels. If something isn't built, write "not built", and write
  "planned" only when the roadmap or decision log says so.
- Plain and short. Use tables where they help. No marketing words.
- Generated sections (the endpoint list in
  [06-API-SPECIFICATION.md](06-API-SPECIFICATION.md)) carry marker comments.
  Regenerate them. Don't edit them by hand.
- Never write secret values. Name the variable instead.
