# Learning log

One entry per module: what I learned, what failed, what to improve.

## Phase 0: environment + data layer (2026-11-03)

- **What I built:** a single `NbaClient` (`src/sportslab/data/client.py`) that every `nba_api` call must go through: disk cache in Parquet, >=1 s between real requests, 60 s timeout, retries with exponential backoff.
- **Why one gateway:** stats.nba.com throttles bursts and blocks many cloud IPs. Centralizing the rules means no module can accidentally hammer it, and a finished season is downloaded exactly once.
- **Cache design:** key = endpoint name + sorted params (param order does not matter). Cache hits skip the rate limit. Writes go to a temp dir and are renamed, so a crash never leaves a half-written entry. `max_age` lets recent-data modules (daily dashboard) expire entries; finished seasons never expire.
- **Testing without network:** time (`sleep`, `clock`) and the endpoint are injected, so rate-limit and backoff behavior is tested deterministically in milliseconds. One separate `-m network` smoke test hits the real API (`LeagueGameLog` 2023-24: 2460 team-game rows = 1230 games).
- **What failed:** I set `requires-python >=3.11`, but the latest numpy needs 3.12+. uv caught it at resolve time. Floor is now 3.12. Also `winget` does not exist on this Windows 10 LTSC; uv installed with `pip install --user uv` and runs as `python -m uv`.
- **To improve:** result sets are stored by index (`0.parquet`...); name them if an endpoint with duplicate column names shows up. Add one smoke test per new endpoint as modules need them.

## Module 1, part 1: Elo baseline (2026-11-03)

- **Result:** on the untouched 2025-26 test season the tuned Elo (K=24, home advantage 40, carryover 0.7) scores log loss 0.6037 vs 0.6931 for a coin and 0.6875 for "home team wins at the historical rate". Brier 0.208, favorite wins 68%. Calibration is reasonable; extreme bins have too few games to judge.
- **Hypothesis vs. result:** I had no prior ("mi intuicion es X"), so I compared against baselines. Lesson: a baseline is a hypothesis you can always state, even without intuition.
- **Surprise:** home-court advantage fell from ~59% (to 2018-19) to ~55% since 2019-20 (2022-23 at 58% is an exception). The classic Elo home bonus of ~100 points is outdated; validation preferred 40.
- **What failed:** (1) real data has neutral-site games (both teams marked "@"), which broke my "one home team per game" assumption; now flagged and excluded from scoring. (2) The best home advantage sits at the edge of my grid (40 = lowest tried); I should have included 0 and 20. (3) I saw validation and test output in the same run, so extending the grid now makes 2025-26 a "second look", not a clean test. (4) I wrote "7 consecutive seasons" for the home-rate drop; 2022-23 contradicts it. I caught it by checking the table before shipping the notebook.
- **Tooling lessons:** Windows PowerShell 5.1 `Get-Content`/`Set-Content` silently corrupted UTF-8 accents in a file; fixed with a byte-exact reverse mapping. Iterating pandas Series row by row was 2x slower than plain lists.
- **Still to do for the module:** logistic regression and XGBoost compared against this Elo (they need pre-game features: rest days, recent form, rating difference), margin-of-victory Elo as an exercise, speed up the 280-run tuning grid (first page load ~25-30 s).
