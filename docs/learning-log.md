# Learning log

One entry per module: what I learned, what failed, what to improve.

## Phase 0: environment + data layer (2026-11-03)

- **What I built:** a single `NbaClient` (`src/sportslab/data/client.py`) that every `nba_api` call must go through: disk cache in Parquet, >=1 s between real requests, 60 s timeout, retries with exponential backoff.
- **Why one gateway:** stats.nba.com throttles bursts and blocks many cloud IPs. Centralizing the rules means no module can accidentally hammer it, and a finished season is downloaded exactly once.
- **Cache design:** key = endpoint name + sorted params (param order does not matter). Cache hits skip the rate limit. Writes go to a temp dir and are renamed, so a crash never leaves a half-written entry. `max_age` lets recent-data modules (daily dashboard) expire entries; finished seasons never expire.
- **Testing without network:** time (`sleep`, `clock`) and the endpoint are injected, so rate-limit and backoff behavior is tested deterministically in milliseconds. One separate `-m network` smoke test hits the real API (`LeagueGameLog` 2023-24: 2460 team-game rows = 1230 games).
- **What failed:** I set `requires-python >=3.11`, but the latest numpy needs 3.12+. uv caught it at resolve time. Floor is now 3.12. Also `winget` does not exist on this Windows 10 LTSC; uv installed with `pip install --user uv` and runs as `python -m uv`.
- **To improve:** result sets are stored by index (`0.parquet`...); name them if an endpoint with duplicate column names shows up. Add one smoke test per new endpoint as modules need them.
