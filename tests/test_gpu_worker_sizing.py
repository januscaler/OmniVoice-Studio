"""GPU worker-sizing policy (#567 crash prevention).

The "Can't reach the local backend" crash wave on 8 GB GPUs was an OOM/CUDA
fault from running 2 concurrent clone jobs (TTS + co-loaded WhisperX ASR) when
the pool was sized to >1 worker on a small card. These pin the sizing policy so
an 8 GB card serializes to a single worker (no contention → no crash) while
larger cards still parallelize, all without needing a GPU.
"""
from services.model_manager import (
    _workers_for_free_vram,
    _GPU_WORKER_CAP,
    _GPU_VRAM_PER_JOB_GB,
)


def test_eight_gb_card_serializes_to_one_worker():
    # An 8 GB card reports ~7 GB free when the pool is sized — must be 1 worker
    # so two concurrent clone jobs can't blow past VRAM (#567/#570/#571/#580+).
    assert _workers_for_free_vram(7.0) == 1
    assert _workers_for_free_vram(6.5) == 1
    # ≤10 GB stays single-worker under the 5 GB/job budget.
    assert _workers_for_free_vram(9.5) == 1


def test_larger_cards_still_parallelize():
    assert _workers_for_free_vram(11.0) == 2   # 12 GB
    assert _workers_for_free_vram(15.0) == 3   # 16 GB
    assert _workers_for_free_vram(23.0) == _GPU_WORKER_CAP  # 24 GB → capped


def test_floor_and_cap():
    # Never zero (a tiny/!-reported free figure still gets one worker)...
    assert _workers_for_free_vram(0.4) == 1
    assert _workers_for_free_vram(0.0) == 1
    # ...and never above the cap, however large the card.
    assert _workers_for_free_vram(256.0) == _GPU_WORKER_CAP


def test_budget_is_conservative_enough_for_the_asr_coload():
    # Guard the constant itself: the co-loaded WhisperX large-v3 (~3 GB) plus
    # TTS (~1.6 GB) means a concurrent clone job needs ~5 GB; a regression back
    # toward 2.5 GB would re-enable the 2-worker-on-8 GB crash.
    assert _GPU_VRAM_PER_JOB_GB >= 5.0


# ── Growing a pool built while the card was busy ──────────────────────────


def test_a_pool_grows_by_what_fits_beside_it_and_never_shrinks():
    from services.model_manager import _grown_workers

    assert _grown_workers(1, 4.0) == 1          # nothing more fits beside one worker
    assert _grown_workers(1, 5.0) == 2
    assert _grown_workers(1, 15.0) == _GPU_WORKER_CAP
    assert _grown_workers(2, 11.0) == _GPU_WORKER_CAP
    assert _grown_workers(_GPU_WORKER_CAP, 40.0) == _GPU_WORKER_CAP
    assert _grown_workers(3, 0.0) == 3          # a busy card never takes workers away


def test_a_pool_with_work_waiting_grows_when_the_card_frees_up(monkeypatch):
    import threading

    import services.model_manager as mm

    monkeypatch.delenv("OMNIVOICE_GPU_WORKERS", raising=False)
    monkeypatch.setattr(mm, "_pick_gpu_workers", lambda: 1)       # built while a render held the card
    pool = mm._ResilientGpuPool()
    release, started = threading.Event(), []

    def job():
        started.append(1)
        release.wait(5)

    futures = [pool.submit(job) for _ in range(3)]
    assert pool.stats()["workers"] == 1                           # no CUDA here: nothing to grow into

    monkeypatch.setattr(mm, "_free_vram_gb", lambda: 12.0)        # the render ended
    pool._grow_checked = float("-inf")
    futures.append(pool.submit(job))
    assert pool.stats()["workers"] == 3                           # 1 + int(12 // 5)
    release.set()
    [f.result(timeout=5) for f in futures]
    pool.shutdown()


def test_a_pinned_pool_never_grows(monkeypatch):
    import services.model_manager as mm

    monkeypatch.setenv("OMNIVOICE_GPU_WORKERS", "1")
    monkeypatch.setattr(mm, "_free_vram_gb", lambda: 30.0)
    pool = mm._ResilientGpuPool()
    pool._pool = mm.ThreadPoolExecutor(max_workers=1)
    pool._queued = 2
    pool._maybe_grow(pool._pool)
    assert pool.stats()["workers"] == 1
    pool.shutdown()
