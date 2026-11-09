from __future__ import annotations

import os
import subprocess
import sys

import pytest
import requests
from huggingface_hub.errors import HfHubHTTPError

from utils import hf_download


def http_error(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return HfHubHTTPError("test download failure", response=response)


def test_rate_limit_retry_preserves_pinned_arguments_and_honors_retry_after(monkeypatch):
    sleeps, calls = [], []
    monkeypatch.setattr(hf_download.time, "sleep", sleeps.append)

    def operation(repo, *, revision):
        calls.append((repo, revision))
        if len(calls) < 3:
            raise http_error(429, "180")
        return "downloaded"

    assert hf_download.retry_hf_call(operation, "org/data", revision="pinned") == "downloaded"
    assert sleeps == [180, 180]
    assert calls == [("org/data", "pinned")] * 3


def test_persistent_rate_limit_is_bounded_and_preserves_original_error(monkeypatch):
    sleeps, calls = [], []
    error = http_error(429)
    monkeypatch.setattr(hf_download.time, "sleep", sleeps.append)

    def operation():
        calls.append(1)
        raise error

    with pytest.raises(HfHubHTTPError) as caught:
        hf_download.retry_hf_call(operation)
    assert caught.value is error
    assert len(calls) == 6
    assert sleeps == [60, 120, 240, 480, 600]


@pytest.mark.parametrize("status", [401, 403, 404])
def test_non_rate_limit_errors_fail_immediately(monkeypatch, status):
    monkeypatch.setattr(hf_download.time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    error = http_error(status)

    def operation():
        raise error

    with pytest.raises(HfHubHTTPError) as caught:
        hf_download.retry_hf_call(operation)
    assert caught.value is error


@pytest.mark.parametrize("value,expected", [(None, 0), ("bad", 0), ("nan", 0), ("-10", 0), ("120", 120)])
def test_retry_after_numeric_or_invalid(value, expected):
    assert hf_download.retry_after_seconds(value) == expected


def test_retry_after_http_date(monkeypatch):
    monkeypatch.setattr(hf_download.time, "time", lambda: 0)
    assert hf_download.retry_after_seconds("Thu, 01 Jan 1970 00:02:00 GMT") == 120


def test_import_preserves_callers_ability_to_configure_hub_environment():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os; import utils.hf_download; "
            "os.environ['HF_HUB_DISABLE_XET'] = '1'; "
            "from huggingface_hub import constants; assert constants.HF_HUB_DISABLE_XET",
        ],
        env={**os.environ, "HF_HUB_DISABLE_XET": "0"},
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
