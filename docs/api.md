# Sports Analytics Lab API (local, read-only)

A small FastAPI service over the models built in the modules. **Educational; not betting advice.**

```bash
python -m uv run uvicorn sportslab.service.api:app --host 127.0.0.1 --port 8000
# interactive docs: http://127.0.0.1:8000/docs
```

## What it is, and deliberately is not

- **It connects to nothing else.** It reads NBA.com through the shared `NbaClient` (rate limit,
  retries, disk cache), computes in memory and answers JSON. No database, no outbound calls to any
  other service, no credentials.
- **Local only.** Binds to `127.0.0.1`, no authentication, CORS not enabled. Do not expose it
  publicly as it is: it would need authentication, rate limiting and a license review of the data
  first (NBA.com terms apply, see the README).
- **Read-only.** Every endpoint is a `GET`.

## Endpoints

| Endpoint | What it returns |
|---|---|
| `GET /health` | Liveness only; loads no data |
| `GET /v1/meta` | Season, data date, whether the data is `stale`, schedule warnings, Elo parameters |
| `GET /v1/ratings` | Elo rating of every team, best first |
| `GET /v1/games?date=YYYY-MM-DD` | Games of a day (default: next day with games) with the pre-game Elo probability; finished games add the score and a `surprise` flag |
| `GET /v1/standings?include_odds=false` | Standings by conference (W-L, last 10, point difference, Elo); `include_odds=true` adds playoff probabilities (3000 simulations) |
| `GET /v1/simulation?n_sims=5000` | Season simulation (500-20000 runs): playoffs, conference finals, finals, title and expected wins with an 80% range |
| `GET /v1/fantasy/projections?team=BOS&limit=50` | Next-game fantasy projections with 50% and 80% intervals |
| `GET /v1/fantasy/projections/{player_id}` | One player's projection (404 if not projected) |

Every probability response carries a `disclaimer`; fantasy responses add that they are conditional
on the player playing (the data has no injury information).

## Behaviour worth knowing

- **Data freshness.** The context is kept in memory for 30 minutes; refreshes go through the
  client (which accepts NBA.com data up to 3 hours old for the season in progress).
- **Failures are explicit.** If a refresh fails but an older copy exists, it is served with
  `"stale": true`. With no data at all the service answers **503** with an explanation (NBA.com
  blocks many cloud IPs: run it from your PC).
- **Validation.** Bad dates and out-of-range `n_sims`/`limit` return **422**; unknown players **404**.
- **Cost.** The first request after startup loads data (seconds to a minute if the disk cache is
  empty); simulations and projections are cached per data version. Measured locally with a warm
  cache: ratings/games/standings under 0.5 s, a 3000-run simulation under 1 s, first projections
  call ~3.5 s.

## Tests

`tests/test_api.py` runs the real routing and logic on synthetic data (no network): validation,
ordering, probabilities that add up, TTL caching, stale fallback and 503.
