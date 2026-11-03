# External data you provide (not in git)

- `kaggle_omarsobhy14_raw.csv`: downloaded from Kaggle `omarsobhy14/nba-players-salaries` (license not confirmed; kept local).
- `salaries.csv`: made from it by `scripts/prepare_salaries.py` with columns `season,player_name,salary_usd`
  (optional `team`, `player_id`, `years_experience`; see `sportslab/data/salaries.py`).

Only 2022-23 covers the whole rotation (100%); 2023-24 covers 79%, 2024-25 52%, 2025-26 29%.
