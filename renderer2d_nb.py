"""
Numba-accelerated 2D trajectory computation
compute_trajectories_nb() runs fully on the JIT-compiled path:
  • One Numba prange loop over all rays (parallel across CPU cores)
  • Each ray returns a dense (MAX_PTS, 2) position buffer + actual length
  • The caller (matplotlib in the GUI) draws the paths from these buffers

This separation keeps all heavy math in Numba while leaving matplotlib
free to handle colours, labels, legends, and interactive display.
"""

import numpy as np
from numba import njit, prange
from physics import rk4_step_nb, adaptive_dt, RS


MAX_PTS = 8000 # Maximum stored positions per ray
DT_BASE = 0.04
MAX_DIST = 60.0


# Single-ray kernel
@njit(cache=True)
def _trace_one(start_x: float, start_y: float,
               dir_x: float, dir_y: float,
               buf: np.ndarray) -> tuple:
    """
    Integrate one photon in the z=0 plane.

    Returns (n_pts, status):
        n_pts — number of valid positions written to buf
        status — 0 = escaped, 1 = captured
    """
    px, py, pz = start_x, start_y, 0.0
    vx, vy, vz = dir_x, dir_y, 0.0

    buf[0, 0] = px
    buf[0, 1] = py

    n = 1
    status = 0

    for _ in range(MAX_PTS - 1):
        r = (px*px + py*py) ** 0.5

        if r < RS * 0.95:
            status = 1
            break
        if r > MAX_DIST:
            status = 0
            break

        dt = adaptive_dt(r, DT_BASE)
        px, py, pz, vx, vy, vz = rk4_step_nb(px, py, pz, vx, vy, vz, dt)

        buf[n, 0] = px
        buf[n, 1] = py
        n += 1

    return n, status


# Parallel multi-ray launcher
@njit(parallel=True, cache=True)
def compute_trajectories_nb(
    starts: np.ndarray, # start positions
    dirs: np.ndarray, # direction vectors
) -> tuple:
    """
    Traces N photons in parallel.

    Returns:
        paths (N, MAX_PTS, 2)  — position buffers
        lengths (N,) — valid point count per ray
        statuses (N,) — 0=escaped, 1=captured
    """
    N = starts.shape[0]
    paths = np.zeros((N, MAX_PTS, 2), dtype=np.float64)
    lengths = np.zeros(N, dtype=np.int64)
    statuses = np.zeros(N, dtype=np.int64)

    for i in prange(N):
        n, s = _trace_one(starts[i, 0], starts[i, 1],
                          dirs[i, 0], dirs[i, 1],
                          paths[i])
        lengths[i]  = n
        statuses[i] = s

    return paths, lengths, statuses


# Build launch arrays for a fan of rays
def make_ray_fan(n_rays: int, b_min: float, b_max: float,
                 start_x: float = -22.0) -> tuple:
    """
    Creates a horizontal fan of parallel rays coming from the left.

    Args:
        n_rays — number of rays
        b_min — minimum impact parameter (y-offset)
        b_max — maximum impact parameter
        start_x — x-position of ray origins

    Returns:
        starts (N,2), dirs (N,2)
    """
    bs = np.linspace(b_min, b_max, n_rays)
    starts = np.column_stack([np.full(n_rays, start_x), bs])

    dirs = np.column_stack([np.ones(n_rays), np.zeros(n_rays)])

    return starts.astype(np.float64), dirs.astype(np.float64)