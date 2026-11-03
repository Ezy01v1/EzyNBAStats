"""Tests for the cached, rate-limited NBA client. No network: endpoints are fakes."""

from datetime import timedelta

import pandas as pd
import pytest

from sportslab.data.client import NbaClient, NbaClientError


class FakeClock:
    """Deterministic time: sleeping advances the clock, nothing really waits."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakeEndpoint:
    """Mimics the nba_api endpoint contract: ctor(timeout=..., **params).get_data_frames()."""

    calls: list[dict] = []
    failures_left = 0

    def __init__(self, timeout: int, **params) -> None:
        type(self).calls.append({"timeout": timeout, **params})
        if type(self).failures_left > 0:
            type(self).failures_left -= 1
            raise ConnectionError("boom")
        self._params = params

    def get_data_frames(self) -> list[pd.DataFrame]:
        season = self._params.get("season", "none")
        return [
            pd.DataFrame({"season": [season, season], "pts": [100, 110]}),
            pd.DataFrame({"team": ["BOS"], "wins": [64]}),
        ]


@pytest.fixture(autouse=True)
def reset_fake():
    FakeEndpoint.calls = []
    FakeEndpoint.failures_left = 0


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def client(tmp_path, clock) -> NbaClient:
    return NbaClient(
        cache_dir=tmp_path,
        min_interval=1.0,
        timeout=90,
        max_attempts=3,
        backoff_base=2.0,
        sleep=clock.sleep,
        clock=clock.time,
    )


def test_roundtrip_returns_all_result_sets(client):
    frames = client.fetch(FakeEndpoint, season="2023-24")
    assert len(frames) == 2
    assert list(frames[0].columns) == ["season", "pts"]
    assert frames[1].loc[0, "team"] == "BOS"


def test_same_request_hits_network_once(client):
    first = client.fetch(FakeEndpoint, season="2023-24")
    second = client.fetch(FakeEndpoint, season="2023-24")
    assert len(FakeEndpoint.calls) == 1
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_param_order_does_not_change_cache_key(client):
    client.fetch(FakeEndpoint, season="2023-24", league="00")
    client.fetch(FakeEndpoint, league="00", season="2023-24")
    assert len(FakeEndpoint.calls) == 1


def test_different_params_are_different_requests(client):
    client.fetch(FakeEndpoint, season="2023-24")
    client.fetch(FakeEndpoint, season="2024-25")
    assert len(FakeEndpoint.calls) == 2


def test_cache_survives_new_client_instance(tmp_path, clock):
    kwargs = {"cache_dir": tmp_path, "sleep": clock.sleep, "clock": clock.time}
    NbaClient(**kwargs).fetch(FakeEndpoint, season="2023-24")
    NbaClient(**kwargs).fetch(FakeEndpoint, season="2023-24")
    assert len(FakeEndpoint.calls) == 1


def test_timeout_is_forwarded_to_endpoint(client):
    client.fetch(FakeEndpoint, season="2023-24")
    assert FakeEndpoint.calls[0]["timeout"] == 90


def test_waits_between_consecutive_network_requests(client, clock):
    client.fetch(FakeEndpoint, season="2022-23")
    clock.now += 0.25  # only 0.25 s passed since the last request
    client.fetch(FakeEndpoint, season="2023-24")
    assert clock.sleeps == [pytest.approx(0.75)]


def test_no_wait_when_enough_time_has_passed(client, clock):
    client.fetch(FakeEndpoint, season="2022-23")
    clock.now += 5
    client.fetch(FakeEndpoint, season="2023-24")
    assert clock.sleeps == []


def test_cache_hits_do_not_consume_rate_limit(client, clock):
    client.fetch(FakeEndpoint, season="2022-23")
    client.fetch(FakeEndpoint, season="2022-23")  # cache hit
    assert clock.sleeps == []


def test_retries_with_exponential_backoff_then_succeeds(client, clock):
    FakeEndpoint.failures_left = 2
    frames = client.fetch(FakeEndpoint, season="2023-24")
    assert len(frames) == 2
    assert len(FakeEndpoint.calls) == 3
    # backoff 2s then 4s; the rate-limit gap is already covered by those sleeps
    assert clock.sleeps == [pytest.approx(2.0), pytest.approx(4.0)]


def test_gives_up_after_max_attempts_and_caches_nothing(client):
    FakeEndpoint.failures_left = 99
    with pytest.raises(NbaClientError, match="2023-24"):
        client.fetch(FakeEndpoint, season="2023-24")
    assert len(FakeEndpoint.calls) == 3  # max_attempts
    FakeEndpoint.failures_left = 0
    client.fetch(FakeEndpoint, season="2023-24")  # a later success is not poisoned by the failure
    assert len(FakeEndpoint.calls) == 4


def test_refresh_forces_a_new_request(client):
    client.fetch(FakeEndpoint, season="2023-24")
    client.fetch(FakeEndpoint, season="2023-24", refresh=True)
    assert len(FakeEndpoint.calls) == 2


def test_refresh_is_not_part_of_the_cache_key_or_endpoint_params(client):
    client.fetch(FakeEndpoint, season="2023-24", refresh=True)
    assert "refresh" not in FakeEndpoint.calls[0]


def test_max_age_expires_stale_cache(client, clock):
    client.fetch(FakeEndpoint, season="2023-24")
    # same wall-clock "now": still fresh
    client.fetch(FakeEndpoint, season="2023-24", max_age=timedelta(hours=1))
    assert len(FakeEndpoint.calls) == 1
    # entries with max_age=0 are always stale
    client.fetch(FakeEndpoint, season="2023-24", max_age=timedelta(0))
    assert len(FakeEndpoint.calls) == 2
