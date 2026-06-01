# TradingAgents — Project Guide

Multi-agent LLM financial trading analysis. A Python/LangGraph agent pipeline
exposed over a FastAPI HTTP API, with a Next.js web frontend.

## Architecture

```
TradingAgents/
├── tradingagents/          # Core multi-agent graph (LangGraph) — analysts, researchers, trader, risk
├── cli/                    # Typer CLI (tradingagents console script)
├── backend/                # Web/server layer
│   ├── server.py           # FastAPI HTTP API + legacy HTML routes (port 8000)
│   ├── worker.py           # Subprocess worker invoked per-analysis (isolates ChromaDB state)
│   └── templates/          # Legacy vanilla HTML UI (dashboard/history/analysis_viewer), kept as fallback
├── frontend/               # Next.js 15 (App Router) + shadcn/ui web app (port 3000)
├── data/                   # Runtime data
│   ├── saved_tickers.json  # Persisted watchlist (tracked)
│   ├── eval_results/        # Per-ticker analysis logs (full_states_log_<date>.json) — gitignored
│   ├── analysis_results/    # Flat per-run result snapshots — gitignored
│   └── results/             # Legacy result dir — gitignored
├── main.py                 # Example script: run the graph directly
├── pyproject.toml / setup.py  # Packaging (root layout — keep here)
└── README.md / LICENSE
```

The frontend is the primary UI. The three legacy HTML files in
`backend/templates/` and their FastAPI `HTMLResponse` routes (`/`, `/history`,
`/analysis`) remain served by the backend as a no-build fallback and are
intentionally left unchanged.

`backend/server.py` resolves all paths (templates, worker, data dirs) relative
to its own location and adds the repo root to `sys.path`, so it runs correctly
regardless of the current working directory.

## Running

Backend (terminal 1):

```bash
# from repo root, with .venv active and .env populated (OPENAI_API_KEY, ALPHA_VANTAGE_API_KEY)
python backend/server.py        # http://localhost:8000
```

Frontend (terminal 2):

```bash
cd frontend
npm install
npm run dev                     # http://localhost:3000
```

The frontend calls the backend through a Next.js rewrite: requests to
`/api/*` are proxied to `http://localhost:8000/*` (see `frontend/next.config.ts`).
Override the target with `API_TARGET` if the backend runs elsewhere.

Production build check: `cd frontend && npm run build`.

## Backend API contract

Consumed by `frontend/src/lib/api.ts`; types in `frontend/src/lib/types.ts`.

| Method | Path                        | Purpose                                              |
| ------ | --------------------------- | ---------------------------------------------------- |
| GET    | `/tickers`                  | `{ tickers: string[] }`                              |
| POST   | `/tickers`                  | Body `{ tickers: string[] }` → persists watchlist    |
| POST   | `/analyze`                  | Run one analysis → `{ ticker, date, decision, status }` |
| GET    | `/results`                  | `{ results: { [ticker]: { date, decision }[] } }`    |
| GET    | `/results/{ticker}/{date}`  | Full analysis state keyed by date                    |
| POST   | `/clear-memory`             | Reset ChromaDB between batch runs                    |
| GET    | `/health`                   | Liveness                                             |

`decision` in `/analyze` may be a dict or string. The detail endpoint returns
`{ [dateKey]: AnalysisState }` where `AnalysisState` includes the analyst
reports, `investment_debate_state`, and `risk_debate_state`.

## Frontend conventions

- Next.js App Router, TypeScript, server-friendly client components.
- shadcn/ui (new-york) + Radix primitives in `src/components/ui`. Feature
  components live one level up in `src/components`.
- Tailwind with HSL CSS-variable tokens; semantic signal colors `buy`/`sell`/`hold`.
- Theming via `next-themes` (default dark, system-aware).
- All financial/numeric text uses the `tabular` class (tabular numerals) and
  the mono font for tickers, dates, and prices.
- User feedback via `sonner` toasts — never `alert()`/`confirm()`.
- Markdown reports render through `react-markdown` + `remark-gfm`, not string replace.

See `frontend/DESIGN.md` for the design system and `DECISIONS.md` for the
rationale behind key choices.
