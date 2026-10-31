"""``POST /separate-audio`` — one recording split into its vocals and background.

A member-facing (consumption) route: local clients and authenticated remote
consumers — API key, share PIN, trusted network — may call it, exactly as they
call the dub routes. It is deliberately NOT on the admin-gated system router
that ``POST /clean-audio`` lives on, so an application backend that separates
recordings (for example to keep a shot's room tone under a voice-over) needs no
credential that also opens ``/system/*``.

The response is a ZIP (``application/zip``) holding ``vocals.wav`` and
``background.wav``. When Demucs cannot split the recording the route answers
with an error that says so — it never returns the input as a stem.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import uuid
import zipfile

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from api.dependencies import require_consumer
from core.config import OUTPUTS_DIR
from services.ffmpeg_utils import find_ffmpeg, run_ffmpeg
from services.separation import DEMUCS_MODEL, SeparationError, separate_two_stems

router = APIRouter()
logger = logging.getLogger("omnivoice.separation")

#: Audio containers accepted, as for audio-only dubbing (``/dub/upload``).
AUDIO_EXTS = frozenset({".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"})
#: Largest upload accepted: about 46 minutes of 24 kHz mono 16-bit WAV. Streamed
#: to disk in bounded chunks, so an oversized body is never held in memory.
MAX_UPLOAD_BYTES = 128 * 1024 * 1024
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_CONVERT_TIMEOUT_S = 120.0

#: How a Demucs failure is answered: status and a fixed public message.
_FAILURES = {
    "unavailable": (503, ("Demucs is not installed on this server, so the recording "
                          "could not be separated.")),
    "timeout": (504, ("Demucs did not finish separating the recording in time; "
                      "nothing was separated.")),
    "failed": (500, "Demucs could not separate the recording; nothing was separated."),
    "no_output": (500, "Demucs could not separate the recording; nothing was separated."),
}


async def _save_upload(audio: UploadFile, destination: str) -> int:
    total = 0
    out = await asyncio.to_thread(open, destination, "wb")
    try:
        while True:
            chunk = await audio.read(_UPLOAD_CHUNK_BYTES)
            if not chunk:
                return total
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"Audio is too large (maximum {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).",
                )
            await asyncio.to_thread(out.write, chunk)
    finally:
        out.close()


async def _to_wav(source: str, destination: str, *, sample_rate: int | None = None,
                  channels: int | None = None) -> bool:
    """Decode ``source`` to PCM WAV (optionally resampled / remixed); whether it worked."""
    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", source, "-vn", "-acodec", "pcm_s16le"]
    if sample_rate:
        cmd += ["-ar", str(sample_rate)]
    if channels:
        cmd += ["-ac", str(channels)]
    cmd.append(destination)
    try:
        rc, _, _ = await run_ffmpeg(cmd, timeout=_CONVERT_TIMEOUT_S)
    except TimeoutError:
        return False
    return rc == 0 and os.path.isfile(destination) and os.path.getsize(destination) > 44


def _zip_stems(vocals: str, background: str, destination: str) -> None:
    # WAV barely compresses; storing keeps the CPU free for the next separation.
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.write(vocals, "vocals.wav")
        archive.write(background, "background.wav")


@router.post("/separate-audio", dependencies=[Depends(require_consumer)])
async def separate_audio(
    audio: UploadFile = File(...),  # noqa: B008 — FastAPI's declaration idiom
    sample_rate: int | None = Form(None, ge=8000, le=48000),
    channels: int | None = Form(None, ge=1, le=2),
):
    """Split an uploaded recording into ``vocals.wav`` and ``background.wav`` (a ZIP).

    ``sample_rate`` / ``channels`` resample both stems; omitted, they keep
    Demucs's own output (44.1 kHz stereo).
    """
    ext = os.path.splitext(audio.filename or "")[1].lower()
    if ext not in AUDIO_EXTS:
        await audio.close()
        raise HTTPException(
            status_code=415,
            detail=f"Separation needs an audio file ({', '.join(sorted(AUDIO_EXTS))}); "
                   f"got '{ext or 'no extension'}'.",
        )
    work = os.path.join(OUTPUTS_DIR, f"_separate_{uuid.uuid4().hex[:12]}")
    os.makedirs(work, exist_ok=True)
    try:
        raw = os.path.join(work, f"upload{ext}")
        try:
            size = await _save_upload(audio, raw)
        finally:
            await audio.close()
        if size == 0:
            raise HTTPException(status_code=400, detail="The uploaded audio is empty.")
        source = os.path.join(work, "input.wav")
        if not await _to_wav(raw, source):
            raise HTTPException(status_code=400,
                                detail="The upload could not be read as audio.")
        try:
            stems = await separate_two_stems(source, work)
        except SeparationError as err:
            status, message = _FAILURES.get(err.reason, _FAILURES["failed"])
            logger.warning("separate-audio: Demucs did not separate the upload (%s: %s)",
                           err.reason, err)
            raise HTTPException(status_code=status, detail=message,
                                headers={"X-Separation-Error": err.reason}) from None
        vocals, background = stems.vocals, stems.background
        if sample_rate or channels:
            vocals, background = (os.path.join(work, "vocals.wav"),
                                  os.path.join(work, "background.wav"))
            for src, dst in ((stems.vocals, vocals), (stems.background, background)):
                if not await _to_wav(src, dst, sample_rate=sample_rate, channels=channels):
                    logger.warning("separate-audio: converting a stem to the requested "
                                   "rate/channels failed")
                    raise HTTPException(
                        status_code=500,
                        detail="The separated stems could not be converted; nothing was returned.",
                        headers={"X-Separation-Error": "conversion"})
        archive = os.path.join(work, "stems.zip")
        await asyncio.to_thread(_zip_stems, vocals, background, archive)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return FileResponse(
        archive, media_type="application/zip", filename="stems.zip",
        headers={"X-Separation-Model": DEMUCS_MODEL},
        background=BackgroundTask(shutil.rmtree, work, ignore_errors=True),
    )
