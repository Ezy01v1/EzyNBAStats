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

## Run the app

```bash
python -m uv run streamlit run app/Home.py
python -m uv run jupyter lab         # notebooks/01_elo_predictor.ipynb
python -m uv run python scripts/elo_experiment.py
```

## Run the API (local, read-only)

```bash
python -m uv run uvicorn sportslab.service.api:app --host 127.0.0.1 --port 8000
```

Docs and design limits (localhost only, no auth, connects to nothing else) in
[docs/api.md](docs/api.md).

## Record and score predictions (Neon)

Predictions are frozen in a Postgres database (a dedicated Neon project, `sports-analytics-lab`)
before the games, then scored against the results: the honest out-of-sample test. Put the
connection string in a local `.env` (git-ignored; see `.env.example`) and run, once a day:

```bash
python -m uv run python scripts/record_predictions.py
python -m uv run python scripts/score_predictions.py
```

A Windows scheduled task, **SportsLab daily predictions**, runs both every day at 11:00 through
`scripts/daily_predictions.cmd` (it catches up when the PC was off; output in
`logs/daily_predictions.log`). It runs on this PC on purpose: stats.nba.com blocks most cloud IPs.

Re-running `record` on the same day adds nothing (the first snapshot of the day wins). Only this
lab's predictions are stored there; nothing is connected to any other app.

