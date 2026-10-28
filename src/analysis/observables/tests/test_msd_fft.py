"""Regression test: the FFT (Fast-Correlation) MSD equals the direct one.

``msd_fft`` is the O(T log T) engine ported from byteff2; it must reproduce the
O(T^2) time-origin reference :func:`msd_time_origin` (and its parallel variant)
to numerical precision on the same unwrapped trajectory. Same MSD => same
Einstein-fit D, so this guards the new default against regressions.

    python -m pytest src/analysis/observables/tests/test_msd_fft.py
    python -m src.analysis.observables.tests.test_msd_fft     # standalone
"""
import numpy as np

from src.analysis.observables.diffusivity import (
    msd_fft, msd_time_origin, msd_time_origin_parallel,
)


def _random_walk(T, M, seed):
    """A diffusive (T, M, 3) trajectory: coordinates grow ~sqrt(t), like a real
    unwrapped MSD input, so the test exercises the regime we actually run in."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.standard_normal((T, M, 3)), axis=0)


def test_fft_matches_direct():
    for seed in (0, 1, 2):
        x = _random_walk(400, 12, seed)
        fft, direct = msd_fft(x), msd_time_origin(x)
        assert np.allclose(fft, direct, rtol=1e-7, atol=1e-6), \
            f"seed {seed}: max abs diff {np.max(np.abs(fft - direct)):.3e}"


def test_fft_matches_parallel():
    x = _random_walk(300, 8, 7)
    assert np.allclose(msd_fft(x), msd_time_origin_parallel(x, n_workers=2),
                       rtol=1e-7, atol=1e-6)


def test_single_particle_and_short():
    # M=1 and a tiny T must not crash and must still match the reference.
    for T, M in ((50, 1), (3, 4), (1, 5)):
        x = _random_walk(T, M, 99)
        assert np.allclose(msd_fft(x), msd_time_origin(x), rtol=1e-7, atol=1e-6)


if __name__ == "__main__":
    test_fft_matches_direct()
    test_fft_matches_parallel()
    test_single_particle_and_short()
    print("OK: msd_fft matches the direct time-origin MSD (incl. parallel + edge cases).")
