"""``POST /separate-audio``: a recording's vocals and background, for consumers.

Pinned here:
- an anonymous remote caller is refused; local clients and authenticated remote
  consumers (trusted network, share PIN, API key) are served;
- the admin gate is not required — a consumer that ``/clean-audio`` refuses is
  served here;
- bad input (wrong type, empty, too large, unreadable, out-of-range options) is a
  clear 4xx and never reaches Demucs;
- a successful split returns a ZIP of ``vocals.wav`` and ``background.wav``;
- when Demucs cannot split it, the route says so instead of echoing the input.
"""

from __future__ import annotations

import io
import shutil
import types
import wave
import zipfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REMOTE = ("10.20.30.40", 50000)


def _wav(seconds: float = 0.5, rate: int = 24000, level: int = 1000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(level.to_bytes(2, "little", signed=True) * int(rate * seconds))
    return out.getvalue()


VOICES = _wav(level=2000)
BACKGROUND = _wav(level=300)


@pytest.fixture
def separation(monkeypatch, tmp_path):
    """The router on a bare app, with ffmpeg and Demucs replaced by fakes."""
    from api.routers import separation as route
    from services import separation as service

    monkeypatch.setattr(route, "OUTPUTS_DIR", str(tmp_path))
    monkeypatch.setattr(route, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.delenv("OMNIVOICE_SERVER_MODE", raising=False)
    monkeypatch.delenv("OMNIVOICE_API_KEY", raising=False)
    monkeypatch.delenv("OMNIVOICE_TRUSTED_NETWORKS", raising=False)
    state = types.SimpleNamespace(convert_calls=[], demucs_calls=0, demucs="ok",
                                  readable=True)

    async def fake_ffmpeg(cmd, timeout=0.0, **_):
        state.convert_calls.append(cmd)
        if not state.readable:
            return 1, b"", b"Invalid data found when processing input"
        src, dst = cmd[cmd.index("-i") + 1], cmd[-1]
        shutil.copyfile(src, dst)
        return 0, b"", b""

    async def fake_demucs(cmd, timeout=0.0, **_):
        state.demucs_calls += 1
        if state.demucs == "fail":
            return 1, b"", b"RuntimeError: CUDA out of memory"
        if state.demucs == "timeout":
            raise TimeoutError
        out_dir, source = cmd[cmd.index("-o") + 1], cmd[cmd.index("-o") - 1]
        stem_dir = Path(out_dir) / service.DEMUCS_MODEL / "input"
        stem_dir.mkdir(parents=True)
        if state.demucs != "silent":
            (stem_dir / "vocals.wav").write_bytes(VOICES)
            (stem_dir / "no_vocals.wav").write_bytes(BACKGROUND)
        assert source.endswith("input.wav")
        return 0, b"", b""

    monkeypatch.setattr(route, "run_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(service, "run_ffmpeg", fake_demucs)
    monkeypatch.setattr(service, "demucs_installed", lambda: state.demucs != "missing")
    monkeypatch.setattr(service, "get_best_device", lambda: "cpu")

    app = FastAPI()
    app.include_router(route.router)
    app.state.network_share = types.SimpleNamespace(pin=None)
    state.app = app
    state.tmp = tmp_path
    return state


def _post(app, client=REMOTE, *, body=None, filename="take.wav", headers=None, data=None):
    return TestClient(app, client=client).post(
        "/separate-audio", headers=headers or {}, data=data or {},
        files={"audio": (filename, VOICES if body is None else body, "audio/wav")})


def _stems(res) -> dict[str, bytes]:
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(res.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_an_anonymous_remote_caller_is_refused(separation):
    res = _post(separation.app)
    assert res.status_code == 403
    assert separation.demucs_calls == 0


def test_a_trusted_network_consumer_is_served_without_any_admin_credential(
        separation, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_TRUSTED_NETWORKS", "10.0.0.0/8")
    assert _stems(_post(separation.app)) == {"vocals.wav": VOICES, "background.wav": BACKGROUND}


def test_the_admin_gate_is_not_required(separation, monkeypatch):
    """The same consumer that /clean-audio's admin gate refuses is served here."""
    from api.routers import system

    monkeypatch.setenv("OMNIVOICE_TRUSTED_NETWORKS", "10.0.0.0/8")
    admin = FastAPI()
    admin.include_router(system.router)
    refused = TestClient(admin, client=REMOTE).post(
        "/clean-audio", files={"audio": ("take.wav", VOICES, "audio/wav")})
    assert refused.status_code == 403
    assert _post(separation.app).status_code == 200


def test_a_share_pin_guest_is_served(separation):
    separation.app.state.network_share.pin = "246810"
    assert _post(separation.app).status_code == 403
    res = _post(separation.app, headers={"X-OmniVoice-Pin": "246810"})
    assert set(_stems(res)) == {"vocals.wav", "background.wav"}


def test_the_api_key_is_accepted_in_server_mode(separation, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_SERVER_MODE", "1")
    monkeypatch.setenv("OMNIVOICE_API_KEY", "a-long-server-key")
    assert _post(separation.app, headers={"Authorization": "Bearer wrong"}).status_code == 403
    res = _post(separation.app, headers={"Authorization": "Bearer a-long-server-key"})
    assert res.status_code == 200


def test_a_local_client_is_served(separation):
    assert _post(separation.app, client=("127.0.0.1", 50000)).status_code == 200


@pytest.mark.parametrize("kind, status", [
    ({"filename": "take.txt"}, 415),
    ({"filename": "take"}, 415),
    ({"body": b""}, 400),
    ({"data": {"sample_rate": "100"}}, 422),
    ({"data": {"channels": "6"}}, 422),
])
def test_bad_input_is_refused_before_demucs_runs(separation, kind, status):
    res = _post(separation.app, client=("127.0.0.1", 50000), **kind)
    assert res.status_code == status, res.text
    assert separation.demucs_calls == 0


def test_an_upload_over_the_limit_is_refused(separation, monkeypatch):
    from api.routers import separation as route

    monkeypatch.setattr(route, "MAX_UPLOAD_BYTES", 1024)
    res = _post(separation.app, client=("127.0.0.1", 50000), body=_wav(seconds=1.0))
    assert res.status_code == 413 and "too large" in res.json()["detail"]
    assert separation.demucs_calls == 0


def test_an_upload_that_is_not_audio_is_refused(separation):
    separation.readable = False
    res = _post(separation.app, client=("127.0.0.1", 50000), body=b"not audio at all")
    assert res.status_code == 400 and "could not be read" in res.json()["detail"]
    assert separation.demucs_calls == 0


def test_stems_are_converted_when_asked(separation):
    res = _post(separation.app, client=("127.0.0.1", 50000),
                data={"sample_rate": "24000", "channels": "1"})
    assert set(_stems(res)) == {"vocals.wav", "background.wav"}
    stem_converts = [c for c in separation.convert_calls if "-ar" in c]
    assert len(stem_converts) == 2
    assert all(c[c.index("-ar") + 1] == "24000" and c[c.index("-ac") + 1] == "1"
               for c in stem_converts)


@pytest.mark.parametrize("demucs, status, reason", [
    ("fail", 500, "failed"),
    ("silent", 500, "no_output"),
    ("timeout", 504, "timeout"),
    ("missing", 503, "unavailable"),
])
def test_when_demucs_cannot_separate_the_route_says_so_and_returns_no_audio(
        separation, demucs, status, reason):
    separation.demucs = demucs
    res = _post(separation.app, client=("127.0.0.1", 50000))
    assert res.status_code == status
    assert res.headers["x-separation-error"] == reason
    assert res.headers["content-type"].startswith("application/json")
    assert "separat" in res.json()["detail"]
    assert "CUDA" not in res.text  # the process's own words stay in the log


def test_nothing_is_left_behind(separation):
    _stems(_post(separation.app, client=("127.0.0.1", 50000)))
    separation.demucs = "fail"
    _post(separation.app, client=("127.0.0.1", 50000))
    assert not any(p.name.startswith("_separate_") for p in separation.tmp.iterdir())
