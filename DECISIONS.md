# Decisions Log

Best-practice decisions made during the UI/UX revamp, with rationale. Newest
first.

## Upstream sync (2026-06-01)

- **Merged 136 upstream commits** (`TauricResearch/TradingAgents`) into the fork's
  `main`, then merged the `feat/ui-revamp` web layer on top and deleted the
  branch (single-branch personal workflow). The web layer (`backend/`,
  `frontend/`, `data/`) is fork-only and did not conflict; only `.gitignore`
  conflicted (resolved by keeping upstream's comprehensive file + re-appending
  the project-specific block).
- **Pinned langgraph checkpoint deps** in `pyproject.toml`
  (`langgraph-checkpoint>=2.1.0,<4.0.0`, `langgraph-checkpoint-sqlite>=2.0.0,<3.0.0`).
  Why: `langgraph` 1.0.x rejects checkpoint 4.x; an unpinned install crashed on
  import (`allowed_objects` kwarg). See CLAUDE.md "Dependency gotcha".
- **Refreshed the frontend model dropdown** to the GPT-5.x names upstream now
  uses (`gpt-5.4-nano/-mini`, `gpt-5.4`, `gpt-5.5`, `gpt-5.5-pro`), defaulting to
  deep=`gpt-5.5` / quick=`gpt-5.4-mini` to match `default_config.py`. The old
  `gpt-4o*` names were stale. Canonical source: `model_catalog.py`.

## Repository structure

- **Isolated the web layer into `backend/` and runtime data into `data/`.** The
  root was mixing the upstream research package with a bolted-on web server,
  loose HTML, and scattered result dirs. Now: `backend/server.py`,
  `backend/worker.py`, `backend/templates/` (legacy HTML), and
  `data/{saved_tickers.json,eval_results,analysis_results,results}`.
- **Left the Python package and packaging at root.** `tradingagents/`, `cli/`,
  `setup.py`, and `pyproject.toml` stay put — `find_packages()`, the
  `tradingagents` console script, and editable installs depend on the root
  layout. Moving them would break imports for no benefit.
- **`server.py` resolves paths from `__file__`, not CWD**, and inserts the repo
  root into `sys.path`, so `python backend/server.py` works from anywhere. The
  per-analysis subprocess runs with `cwd=PROJECT_ROOT` so the worker can import
  `tradingagents`.
- **Fixed the Windows cp1252 crash** by forcing UTF-8 stdout/stderr in
  `server.py` (the `✓` startup log previously raised `UnicodeEncodeError`).
- **Renamed for clarity:** `fastapi_server.py → backend/server.py`,
  `trading_worker.py → backend/worker.py`. Used `git mv` to preserve history.

## Stack & structure

- **Next.js 15 (App Router) + TypeScript** replaces the stale Vite + React 19
  scaffold in `frontend/`. Honors the requested stack and gives file-based
  routing, RSC, and a first-class production build/lint gate.
- **shadcn/ui (new-york) over Radix + Tailwind.** Components are copied into
  `src/components/ui` (owned, not a black-box dependency), composable and themeable.
- **`src/` layout with `@/*` path alias.** Standard Next.js convention; keeps
  feature components (`src/components`) separate from primitives
  (`src/components/ui`) and logic (`src/lib`).
- **Scoped the root `.gitignore` `src/` rule to `/src/`.** The unanchored `src/`
  rule (for the Python layout) was silently ignoring `frontend/src/`; anchoring
  it fixes tracking without affecting the Python package.

## Backend integration

- **Next.js `rewrites()` proxy `/api/* → :8000` instead of CORS-coupled direct
  calls.** Keeps the frontend origin-relative (no hardcoded host), avoids CORS
  in dev, and makes the backend target configurable via `API_TARGET`.
- **FastAPI backend and its data endpoints reused unchanged.** The revamp is
  UI-only; no API contract changes were needed.
- **Legacy HTML pages and routes left in place.** Zero-risk fallback; the new
  app lives alongside and can fully replace them once verified.

## Design system (applied ui-ux-pro-max reasoning)

- **Modern fintech-terminal aesthetic**, dark-default, documented in
  `frontend/DESIGN.md`. Removes the legacy purple-gradient "AI slop".
- **HSL CSS-variable tokens + semantic `buy`/`sell`/`hold` colors.** Theme-able,
  AA-contrast in both modes.
- **Tabular numerals + mono font for all financial figures** so numbers align
  and don't shift on update.
- **Signals never rely on color alone** — every decision badge pairs color with
  text and an icon (accessibility + colorblind safety).

## UX & quality

- **`sonner` toasts replace `alert()`/`confirm()`** for non-blocking, accessible
  feedback.
- **`react-markdown` + `remark-gfm` replace the hand-rolled regex markdown
  parser**, which mishandled tables, nesting, and was an XSS risk.
- **`react-hook-form` + `zod`** for typed, validated analysis config.
- **`Skeleton` loaders that mirror final layout** instead of bare spinners,
  reducing layout shift.
- **`prefers-reduced-motion` honored globally.**
- **Production build is the quality gate** (`npm run build` type-checks + lints);
  kept green before each commit.

## Process

- **Incremental, scoped commits** on a `feat/ui-revamp` branch (scaffold →
  dashboard/shared → history/analysis → docs/polish), each building cleanly.
