"""The single gateway to nba_api.

Every request to stats.nba.com goes through ``NbaClient.fetch``:

* disk cache (one Parquet file per result set) so the same request is never made twice,
* a minimum pause between real network requests (cache hits are free),
* a generous timeout,
* retries with exponential backoff.

If all attempts fail it raises ``NbaClientError``; it never returns made-up data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from sportslab.config import RAW_DIR

logger = logging.getLogger(__name__)

_META_FILE = "meta.json"


class NbaClientError(RuntimeError):
    """A request kept failing after all retries (blocked IP, no internet, endpoint changed...)."""


class NbaClient:
    def __init__(
        self,
        cache_dir: Path = RAW_DIR,
        *,
        min_interval: float = 1.0,
        timeout: int = 60,
        max_attempts: int = 4,
        backoff_base: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def fetch(
        self,
        endpoint: type,
        *,
        refresh: bool = False,
        max_age: timedelta | None = None,
        **params: Any,
    ) -> list[pd.DataFrame]:
        """Return every result set of ``endpoint(**params)`` as DataFrames.

        ``refresh`` forces a new request; ``max_age`` treats older cache entries as stale
        (None = cache never expires, right for finished seasons).
        """
        entry = self._entry_dir(endpoint, params)
        if not refresh:
            cached = self._read_cache(entry, max_age)
            if cached is not None:
                return cached

        frames = self._request_with_retries(endpoint, params)
        self._write_cache(entry, endpoint, params, frames)
        return frames

    # -- network -----------------------------------------------------------------------

    def _request_with_retries(self, endpoint: type, params: dict[str, Any]) -> list[pd.DataFrame]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            self._respect_rate_limit()
            self._last_request_at = self._clock()
            try:
                return list(endpoint(timeout=self.timeout, **params).get_data_frames())
            except Exception as exc:  # nba_api raises requests/JSON errors of many kinds
                last_error = exc
                logger.warning(
                    "%s attempt %d/%d failed: %r",
                    endpoint.__name__,
                    attempt,
                    self.max_attempts,
                    exc,
                )
                if attempt < self.max_attempts:
                    self._sleep(self.backoff_base**attempt)
        raise NbaClientError(
            f"{endpoint.__name__}({params}) failed after {self.max_attempts} attempts: "
            f"{last_error!r}. stats.nba.com often blocks cloud IPs and rapid bursts; "
            "run from your own PC or load a CSV alternative."
        ) from last_error

    def _respect_rate_limit(self) -> None:
        if self._last_request_at is None:
            return
        wait = self.min_interval - (self._clock() - self._last_request_at)
        if wait > 0:
            self._sleep(wait)

    # -- cache -------------------------------------------------------------------------

    def _entry_dir(self, endpoint: type, params: dict[str, Any]) -> Path:
        key = json.dumps(params, sort_keys=True, default=str)
        digest = hashlib.sha1(f"{endpoint.__name__}|{key}".encode()).hexdigest()[:16]
        return self.cache_dir / endpoint.__name__ / digest

    def _read_cache(self, entry: Path, max_age: timedelta | None) -> list[pd.DataFrame] | None:
        meta_path = entry / _META_FILE
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if max_age is not None:
            age = datetime.now(UTC) - datetime.fromisoformat(meta["fetched_at"])
            if age >= max_age:
                return None
        return [pd.read_parquet(entry / f"{i}.parquet") for i in range(meta["n_frames"])]

    def _write_cache(
        self, entry: Path, endpoint: type, params: dict[str, Any], frames: list[pd.DataFrame]
    ) -> None:
        # Write into a temp dir and rename, so a crash never leaves a half-written entry.
        tmp = entry.with_name(entry.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        for i, frame in enumerate(frames):
            frame.to_parquet(tmp / f"{i}.parquet", index=False)
        meta = {
            "endpoint": endpoint.__name__,
            "params": json.loads(json.dumps(params, default=str)),
            "fetched_at": datetime.now(UTC).isoformat(),
            "n_frames": len(frames),
        }
        (tmp / _META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        shutil.rmtree(entry, ignore_errors=True)
        tmp.rename(entry)
