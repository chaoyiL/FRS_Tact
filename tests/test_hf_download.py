from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

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


@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "Consistency check failed: file should be of size 20 but has size 30 (episode.parquet).",
            "data/episode.parquet",
        ),
        (
            "Consistency check failed: file should be of size 20 but has size 30 (data/episode.parquet).",
            "data/episode.parquet",
        ),
        ("Consistency check failed: file should be of size 99 but has size 30 (episode.parquet).", None),
        ("Consistency check failed: file should be of size 20 but has size 30 (other.parquet).", None),
        ("[Errno 122] Disk quota exceeded", None),
    ],
)
def test_consistency_error_resolves_only_selected_file(message, expected):
    assert hf_download.consistency_error_file(OSError(message), {"data/episode.parquet": 20}) == expected


def test_ambiguous_consistency_error_does_not_choose_a_file():
    error = OSError("Consistency check failed: file should be of size 20 but has size 30 (episode.parquet).")
    assert (
        hf_download.consistency_error_file(error, {"a/episode.parquet": 20, "b/episode.parquet": 20}) is None
    )


def test_forced_repair_retries_only_bad_file_and_keeps_revision(tmp_path, monkeypatch):
    import huggingface_hub

    calls = []
    (tmp_path / "good.parquet").write_bytes(b"keep")

    def download(**kwargs):
        calls.append(kwargs)
        assert kwargs["force_download"] is True
        assert kwargs["filename"] == "data/bad.parquet"
        assert kwargs["revision"] == "pinned"
        if len(calls) < 3:
            raise OSError("Consistency check failed: file should be of size 4 but has size 8 (bad.parquet).")
        path = Path(kwargs["local_dir"]) / kwargs["filename"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"good")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    hf_download.force_download_hf_file(
        "org/data",
        "data/bad.parquet",
        repo_type="dataset",
        revision="pinned",
        local_dir=tmp_path,
        expected_size=4,
    )
    assert len(calls) == 3
    assert (tmp_path / "good.parquet").read_bytes() == b"keep"


@pytest.mark.parametrize("failure,expected_calls", [("consistency", 3), ("range", 3), ("quota", 1)])
def test_forced_repair_is_bounded_and_does_not_retry_disk_quota(
    tmp_path, monkeypatch, failure, expected_calls
):
    import huggingface_hub

    calls = []
    if failure == "consistency":
        error = OSError("Consistency check failed: file should be of size 4 but has size 8 (bad.parquet).")
    elif failure == "range":
        error = http_error(416)
    else:
        error = OSError(122, "Disk quota exceeded")

    def download(**kwargs):
        calls.append(kwargs)
        raise error

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(OSError) as caught:
        hf_download.force_download_hf_file(
            "org/data",
            "bad.parquet",
            repo_type="dataset",
            revision="pinned",
            local_dir=tmp_path,
            expected_size=4,
        )
    assert caught.value is error
    assert len(calls) == expected_calls


def test_range_failure_restarts_known_file_without_parsing_signed_url(tmp_path, monkeypatch):
    import huggingface_hub

    flags = []
    error = http_error(416)
    error.response.url = "https://cdn.example.test/opaque-signed-object"

    def download(**kwargs):
        flags.append(kwargs["force_download"])
        assert kwargs["filename"] == "bad.parquet"
        assert kwargs["revision"] == "pinned"
        if not kwargs["force_download"]:
            raise error
        (tmp_path / kwargs["filename"]).write_bytes(b"good")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    hf_download.download_hf_file(
        "org/data", "bad.parquet", repo_type="dataset", revision="pinned", local_dir=tmp_path, expected_size=4
    )
    assert flags == [False, True]


@pytest.mark.parametrize("status", [403, 404, 429])
def test_single_file_download_propagates_other_http_errors_without_forcing(tmp_path, monkeypatch, status):
    import huggingface_hub

    flags = []
    error = http_error(status)

    def download(**kwargs):
        flags.append(kwargs["force_download"])
        raise error

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    with pytest.raises(HfHubHTTPError) as caught:
        hf_download.download_hf_file(
            "org/data",
            "bad.parquet",
            repo_type="dataset",
            revision="pinned",
            local_dir=tmp_path,
            expected_size=4,
        )
    assert caught.value is error
    assert flags == [False]
