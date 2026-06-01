# TradingAgents — Design System

Source of truth for the web UI. Derived by applying the ui-ux-pro-max-skill
reasoning (industry match → style → palette → typography → anti-pattern
filtering → a11y checklist) to a financial trading-analysis product.

## 1. Product & industry match

A data-dense, decision-oriented tool for reviewing multi-agent trading
analyses. Industry: **fintech / trading terminal**. Users scan numbers, signals
(BUY/SELL/HOLD), and long-form agent reports, then drill in. This calls for a
calm, professional, information-first interface — not a marketing aesthetic.

## 2. Style

**Modern fintech terminal** (Linear × Bloomberg). Flat surfaces, hairline
borders, restrained elevation, fast and subtle motion. Content density over
decoration.

## 3. Color

HSL CSS variables in `src/app/globals.css`, light + dark.

- **Base:** neutral slate. Dark default (`--background: 240 10% 6%`), light
  available via toggle.
- **Semantic signals** (never the only cue — always paired with text + icon):
  - `buy` — green (`142 71% 38%` light / `142 64% 48%` dark)
  - `sell` — red (`0 72% 51%` / `0 72% 58%`)
  - `hold` — amber (`38 92% 45%` / `38 92% 55%`)
  - `UNKNOWN` — muted neutral
- **Primary action:** near-black in light, near-white in dark (high-contrast,
  neutral — accent color is reserved for data signals, not chrome).

## 4. Typography

- **UI / body:** Geist Sans.
- **Numbers, tickers, dates, prices, code:** Geist Mono with `tabular`
  (tabular-nums) so figures align in columns and don't jitter on update.
- Scale: page title `text-2xl`, card title `text-base`/`font-semibold`, body
  `text-sm`, metadata `text-xs uppercase tracking-wide text-muted-foreground`.

## 5. Layout & spacing

- Centered `container`, max-width 1400px, generous `py-8`.
- Cards (`rounded-xl border shadow-sm`) as the primary content unit.
- Responsive grids: dashboard `lg:grid-cols-2`, history stats up to
  `lg:grid-cols-5`.

## 6. Motion

- Transitions ≤ 200ms, ease-out. Accordion uses Radix height animation.
- Loading uses `Skeleton` placeholders that mirror final layout, plus a single
  spinner only for long-running analysis runs.
- `prefers-reduced-motion` honored globally in `globals.css`.

## 7. Components

shadcn/ui (new-york) primitives in `src/components/ui`. Feature components:
`site-header`, `theme-toggle`, `ticker-manager`, `analysis-form`,
`analysis-result`, `batch-progress`, `stat-card`, `report-section`,
`decision-badge`, `markdown`.

## 8. Anti-patterns removed (from the legacy HTML UI)

- ❌ Purple `#667eea → #764ba2` gradient backgrounds + heavy drop shadows.
- ❌ Text-shadow headings and emoji-as-icon nav.
- ❌ Native `alert()` / `confirm()` dialogs → replaced with toasts.
- ❌ Hand-rolled regex `markdownToHtml` → replaced with `react-markdown`.
- ❌ Color-only status (now badge text + icon + color).

## 9. Accessibility checklist

- Radix primitives provide focus management, ARIA roles, keyboard nav.
- All inputs have associated `<Label>` / `aria-label`.
- Signals convey meaning via text + icon, not color alone.
- AA contrast verified in both themes for text and signal badges.
- Visible focus rings (`focus-visible:ring-ring`).
- Reduced-motion respected.
