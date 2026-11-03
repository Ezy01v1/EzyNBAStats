# Sports Analytics Lab

A local, modular NBA analytics lab: shared data layer in `src/sportslab`, one Streamlit page per module in `app/`.
Built to learn sports analytics and to be wrapped later with FastAPI for EzySports.

## Setup (Windows)

```bash
python -m pip install --user uv
python -m uv sync
python -m uv run pytest              # unit tests (no network)
python -m uv run pytest -m network   # smoke tests against the real stats.nba.com
python -m uv run ruff check .
```

Run from your own PC: stats.nba.com blocks many cloud IPs and throttles bursts.
All `nba_api` calls go through `sportslab.data.NbaClient` (cache + rate limit + retries).
Raw data lives in `data/raw/` (git-ignored; NBA.com terms apply).
