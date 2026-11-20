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


# ── Growing a pool built while the card was busy (opt-in) ─────────────────


def test_growth_keeps_a_reserve_and_is_capped_at_two_by_default():
    from services.model_manager import _grown_workers

    assert _grown_workers(1, 10.0) == 1         # nothing beyond the 10 GB reserve
    assert _grown_workers(1, 14.9) == 1
    assert _grown_workers(1, 15.0) == 2         # one 5 GB budget fits beside the reserve
    assert _grown_workers(1, 40.0) == 2         # capped at 2 however free the card is
    assert _grown_workers(1, 40.0, cap=4) == _GPU_WORKER_CAP
    assert _grown_workers(3, 0.0) == 3          # a busy card never takes workers away


def _grow_pool(monkeypatch, mm, free):
    monkeypatch.delenv("OMNIVOICE_GPU_WORKERS", raising=False)
    monkeypatch.setattr(mm, "_pick_gpu_workers", lambda: 1)
    monkeypatch.setattr(mm, "_free_vram_gb", lambda: free[0])
    return mm._ResilientGpuPool()


def test_the_default_pool_never_grows(monkeypatch):
    import threading

    import services.model_manager as mm

    monkeypatch.delenv("OMNIVOICE_GPU_POOL_GROW", raising=False)
    free = [40.0]
    pool = _grow_pool(monkeypatch, mm, free)
    release = threading.Event()
    futures = [pool.submit(release.wait, 5) for _ in range(4)]
    pool._grow_checked = float("-inf")
    futures.append(pool.submit(release.wait, 5))
    assert pool.stats()["workers"] == 1
    release.set()
    [f.result(timeout=5) for f in futures]
    pool.shutdown()


def test_an_opted_in_pool_grows_to_two_when_there_is_room_and_a_reserve(monkeypatch):
    import threading

    import services.model_manager as mm

    monkeypatch.setenv("OMNIVOICE_GPU_POOL_GROW", "1")
    free = [8.0]                                          # a render holds the card
    pool = _grow_pool(monkeypatch, mm, free)
    release = threading.Event()
    futures = [pool.submit(release.wait, 5) for _ in range(3)]
    assert pool.stats()["workers"] == 1
    free[0] = 30.0                                        # the render ended
    pool._grow_checked = float("-inf")
    futures.append(pool.submit(release.wait, 5))
    assert pool.stats()["workers"] == 2                   # capped at 2, not 1 + 30 // 5
    release.set()
    [f.result(timeout=5) for f in futures]
    pool.shutdown()


def test_a_grown_pool_stops_growing_when_a_render_loads_and_shrinks_when_idle(monkeypatch):
    import threading
    import time

    import services.model_manager as mm

    monkeypatch.setenv("OMNIVOICE_GPU_POOL_GROW", "1")
    free = [30.0]
    pool = _grow_pool(monkeypatch, mm, free)
    release = threading.Event()
    futures = [pool.submit(release.wait, 5) for _ in range(3)]
    assert pool.stats()["workers"] == 2
    free[0] = 4.0                                         # a render is loading: under the reserve
    pool._grow_checked = float("-inf")
    futures.append(pool.submit(release.wait, 5))
    assert pool.stats()["workers"] == 1                   # back to base: no new worker takes memory
    free[0] = 30.0
    pool._grow_checked = float("-inf")
    pool.submit(lambda: None)                              # regrown
    release.set()
    [f.result(timeout=5) for f in futures]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and pool._pool is not None:
        time.sleep(0.05)
    assert pool._pool is None, "a drained grown pool is dropped; the next submit builds the base size"
    assert pool.submit(lambda: 7).result(timeout=5) == 7
    pool.shutdown()


def test_a_pinned_pool_never_grows(monkeypatch):
    import services.model_manager as mm

    monkeypatch.setenv("OMNIVOICE_GPU_POOL_GROW", "1")
    monkeypatch.setenv("OMNIVOICE_GPU_WORKERS", "1")
    monkeypatch.setattr(mm, "_free_vram_gb", lambda: 30.0)
    pool = mm._ResilientGpuPool()
    pool._pool = mm.ThreadPoolExecutor(max_workers=1)
    pool._queued = 2
    pool._maybe_grow(pool._pool)
    assert pool.stats()["workers"] == 1
    pool.shutdown()
