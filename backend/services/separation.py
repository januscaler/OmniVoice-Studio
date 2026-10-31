"""Two-stem Demucs separation of one recording: its vocals and everything else.

The single place the ``demucs.separate`` command for a standalone recording is
assembled. ``POST /clean-audio`` keeps the vocals of a microphone take;
``POST /separate-audio`` returns both stems. The dub pipeline's prep runs the
same model and stems through its own streaming runner so it can report
progress (``services.dub_pipeline``).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass

from services.ffmpeg_utils import run_ffmpeg
from services.model_manager import get_best_device

DEMUCS_MODEL = "htdemucs"
#: How long one separation may run before it is abandoned (as /clean-audio always allowed).
DEMUCS_TIMEOUT_S = 900.0


class SeparationError(Exception):
    """Demucs did not split the recording.

    ``reason`` is a stable code: ``unavailable`` (Demucs is not installed),
    ``timeout``, ``failed`` (the process exited non-zero) or ``no_output`` (it
    exited cleanly but wrote no stems).
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class TwoStems:
    """Paths of the two WAV stems Demucs wrote."""

    vocals: str
    background: str


def demucs_installed() -> bool:
    return importlib.util.find_spec("demucs") is not None


async def separate_two_stems(source_path: str, out_dir: str, *,
                             timeout: float = DEMUCS_TIMEOUT_S) -> TwoStems:
    """Split ``source_path`` into ``vocals`` and ``no_vocals`` under ``out_dir``.

    Raises :class:`SeparationError` when Demucs cannot; it never hands back the
    input in place of a stem.
    """
    if not demucs_installed():
        raise SeparationError("unavailable", "Demucs is not installed")
    cmd = [sys.executable, "-m", "demucs.separate", "--two-stems", "vocals",
           "-n", DEMUCS_MODEL, "-d", get_best_device(), source_path, "-o", out_dir]
    try:
        rc, _, _ = await run_ffmpeg(cmd, timeout=timeout)
    except TimeoutError as err:
        raise SeparationError("timeout", f"Demucs did not finish within {timeout:.0f} s") from err
    except OSError as err:
        raise SeparationError("failed", f"Demucs could not start ({type(err).__name__})") from err
    if rc != 0:
        raise SeparationError("failed", f"Demucs exited with code {rc}")
    stem_dir = os.path.join(out_dir, DEMUCS_MODEL,
                            os.path.splitext(os.path.basename(source_path))[0])
    stems = TwoStems(vocals=os.path.join(stem_dir, "vocals.wav"),
                     background=os.path.join(stem_dir, "no_vocals.wav"))
    if not (os.path.isfile(stems.vocals) and os.path.isfile(stems.background)):
        raise SeparationError("no_output", "Demucs wrote no stems")
    return stems
