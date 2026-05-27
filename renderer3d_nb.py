"""
Numba-accelerated 3D ray tracer

render_3d_nb() is the only public function.  It:
  1. Builds camera rays (vectorised numpy)
  2. Calls the @njit(parallel=True) pixel loop
  3. Returns a (H, W, 3) float32 image array

All physics, disk intersection, starfield lookup, and Doppler math
run inside a single parallel Numba kernel
"""

import numpy as np
from numba import njit, prange
from physics import rk4_step_nb, adaptive_dt, RS
import math

DT_BASE = 0.5
MAX_STEPS = 2500
ESCAPE_R = 55.0
PI = math.pi


# Starfield — generated once, passed as a plain numpy array
def generate_starfield(seed: int = 42,
                       n_bright: int = 4000,
                       width: int = 1024,
                       height: int = 512) -> np.ndarray:
    """
    Procedural equirectangular star map.
    Returns float32 (H, W, 3) in [0, 1].

    Stars have temperature-based colours:
        hot (t = 1) -> blue-white
        cool (t = 0) -> orange-red
    A faint Milky Way glow is added along the equator.
    """
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # Dim background speckle
    noise = rng.random((height, width)).astype(np.float32)
    img[noise > 0.997] = 0.35

    # Bright stars
    sx = (rng.random(n_bright) * width ).astype(int)
    sy = (rng.random(n_bright) * height).astype(int)
    br = (rng.random(n_bright) * 0.8 + 0.2).astype(np.float32)
    tmp = rng.random(n_bright).astype(np.float32) # temperature 0..1


    r_ch = np.clip(1.5 - tmp, 0, 1)
    g_ch = np.clip(0.8 + 0.4*tmp - (tmp - 0.5)**2, 0, 1)
    b_ch = np.clip(tmp * 1.5, 0, 1)

    for i in range(n_bright):
        # 3x3 grid around the star
        for dy in range(-1, 2):
            for dx in range(-1, 2):

                py_ = (sy[i] + dy) % height
                px_ = (sx[i] + dx) % width

                w = 1.0 if (dy == 0 and dx == 0) else 0.22

                img[py_, px_, 0] = min(img[py_, px_, 0] + br[i]*w*r_ch[i], 1.0) #r
                img[py_, px_, 1] = min(img[py_, px_, 1] + br[i]*w*g_ch[i], 1.0) #g
                img[py_, px_, 2] = min(img[py_, px_, 2] + br[i]*w*b_ch[i], 1.0) #b

    # Milky Way band
    ys = np.arange(height, dtype=np.float32)

    band = np.exp(-((ys - height*0.5)**2) / (height*0.06)**2)

    mw = (rng.random((height, width)).astype(np.float32) *
            band[:, None] * 0.09)

    img += mw[:, :, None]

    return np.clip(img, 0, 1).astype(np.float32)


# Camera
def build_rays(width: int, height: int,
               cam_pos: np.ndarray, look_at: np.ndarray,
               up: np.ndarray, fov_deg: float) -> np.ndarray:
    """Returns (H*W, 3) unit direction vectors (float64)."""
    fwd = look_at - cam_pos
    fwd /= np.linalg.norm(fwd)

    rgt = np.cross(fwd, up)
    rgt /= np.linalg.norm(rgt)

    cup = np.cross(rgt, fwd)

    hw = np.tan(np.radians(fov_deg / 2.0))
    hh = hw / (width / height)

    u = ((np.arange(width)  + 0.5) / width  * 2.0 - 1.0) * hw
    v = ((np.arange(height) + 0.5) / height * 2.0 - 1.0) * hh
    uu, vv = np.meshgrid(u, v[::-1])

    dirs = fwd + uu[..., None]*rgt + vv[..., None]*cup
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    return dirs.reshape(-1, 3).astype(np.float64)


# Numba kernel - sky lookup
@njit(cache=True, inline='always')
def _sky(vx, vy, vz, sf):
    H, W = sf.shape[0], sf.shape[1]

    r  = (vx*vx + vy*vy + vz*vz) ** 0.5

    if r < 1e-12:
        return 0.0, 0.0, 0.0

    dx, dy, dz = vx/r, vy/r, vz/r

    phi   = np.arctan2(dy, dx)
    sin_z = min(1.0, max(-1.0, dz))
    theta = np.arcsin(sin_z)

    u = (phi / (2.0*PI) + 0.5) % 1.0
    v = theta / PI + 0.5

    xi = int(u * W) % W
    yi = int(v * H) % H

    return float(sf[yi, xi, 0]), float(sf[yi, xi, 1]), float(sf[yi, xi, 2])


# Numba kernel — disk colour + Doppler
@njit(cache=True, inline='always')
def _disk_color(r, phi, doppler, r_in, r_out):
    t = (r - r_in) / (r_out - r_in)
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0

    base   = (1.0 - t)**1.8 * 0.88 + 0.12
    turb   = 0.15 * np.sin(7.0*phi + 4.0*t) * np.sin(2.5*t*PI)
    bright = base + turb
    if bright < 0.0: bright = 0.0
    if bright > 1.0: bright = 1.0
    d = doppler
    if d < 0.05: d = 0.05
    if d > 6.0:  d = 6.0
    bright *= d
    if bright < 0.0: bright = 0.0
    if bright > 1.0: bright = 1.0

    if t < 0.15:
        rc = 0.85;  gc = 0.80 + 0.15*(1.0 - t/0.15);  bc = 1.00
    elif t < 0.35:
        tt = (t - 0.15)/0.20
        rc = 1.00;  gc = 0.90 - 0.25*tt;  bc = 0.55 - 0.45*tt
    elif t < 0.60:
        tt = (t - 0.35)/0.25
        rc = 1.00 - 0.1*tt;  gc = 0.55 - 0.25*tt;  bc = 0.10 - 0.08*tt
    else:
        tt = (t - 0.60)/0.40
        rc = 0.85 - 0.4*tt;  gc = 0.25 - 0.2*tt;  bc = 0.02

    def c(x): return max(0.0, min(1.0, x*bright))
    return c(rc), c(gc), c(bc)


@njit(cache=True, inline='always')
def _doppler(hx, hy, cam_px, cam_py):
    r2d = (hx*hx + hy*hy) ** 0.5

    if r2d < 1e-6:
        return 1.0

    v_orb = (0.5 / r2d) ** 0.5
    if v_orb > 0.45: v_orb = 0.45

    rh_x = hx/r2d;  rh_y = hy/r2d
    vdx = -rh_y * v_orb;  vdy = rh_x * v_orb
    tcx = cam_px - hx;  tcy = cam_py - hy
    tcn = (tcx*tcx + tcy*tcy) ** 0.5

    if tcn < 1e-6: return 1.0
    tcx /= tcn;  tcy /= tcn
    beta = vdx*tcx + vdy*tcy

    if beta < -0.90: beta = -0.90
    if beta >  0.90: beta = 0.90

    D = ((1.0 + beta)/(1.0 - beta)) ** 0.5
    return D*D*D


# Main parallel render kernel
@njit(parallel=True, cache=True)
def _render_kernel(
    rays: np.ndarray, # (N, 3)  float64
    cam_px: float,
    cam_py: float,
    cam_pz: float,
    starfield: np.ndarray, # (SH, SW, 3) float32
    disk_on: bool,
    r_in: float,
    r_out: float,
    dt_base:  float,
) -> np.ndarray:
    """
    Parallel pixel loop

    Each pixel is independent, so prange distributes work across all
    CPU cores with no synchronisation overhead (embarrassingly parallel).
    """
    N = rays.shape[0]
    out = np.zeros((N, 3), dtype=np.float32)

    for idx in prange(N):
        px, py, pz = cam_px, cam_py, cam_pz

        vx = rays[idx, 0]
        vy = rays[idx, 1]
        vz = rays[idx, 2]

        rr = 0.0
        gg = 0.0
        bb = 0.0

        outcome = 0 # 0=escape 1=capture 2=disk

        for _ in range(MAX_STEPS):
            r = (px*px + py*py + pz*pz) ** 0.5

            if r < RS * 0.95:
                outcome = 1; break
            if r > ESCAPE_R:
                outcome = 0; break

            dt = adaptive_dt(r, dt_base)
            npx, npy, npz, nvx, nvy, nvz = rk4_step_nb(px, py, pz, vx, vy, vz, dt)

            # Disk crossing test
            if disk_on:
                z0, z1 = pz, npz
                if z0*z1 <= 0.0 and abs(z0 - z1) > 1e-10:
                    t_c  = -z0 / (z1 - z0)

                    hx   = px + t_c*(npx - px)
                    hy   = py + t_c*(npy - py)
                    r_hit = (hx*hx + hy*hy) ** 0.5

                    if r_in <= r_hit <= r_out:
                        phi_hit = np.arctan2(hy, hx)
                        d3 = _doppler(hx, hy, cam_px, cam_py)
                        rr, gg, bb = _disk_color(r_hit, phi_hit, d3, r_in, r_out)
                        outcome = 2
                        break

            px, py, pz = npx, npy, npz
            vx, vy, vz = nvx, nvy, nvz

        if outcome == 0:
            rr, gg, bb = _sky(vx, vy, vz, starfield)
        elif outcome == 1:
            rr = gg = bb = 0.0

        out[idx, 0] = rr
        out[idx, 1] = gg
        out[idx, 2] = bb

    return out

def render_3d(
    width: int,
    height: int,
    cam_pos: np.ndarray,
    look_at: np.ndarray,
    fov_deg: float,
    starfield: np.ndarray,
    disk_on: bool  = True,
    r_in: float = 6.0,
    r_out: float = 24.0,
    dt_base: float = 0.5,
) -> np.ndarray:
    """
    Render a (height, width, 3) float32 image.

    Args:
        width, height — output resolution
        cam_pos — observer position (3,)
        look_at — gaze target (3,)
        fov_deg — horizontal field of view
        starfield — pre-generated star texture
        disk_on — enable accretion disk
        r_in, r_out — disk inner/outer radius
        dt_base — integration step size

    Returns float32 array in [0, 1].
    """
    up   = np.array([0.0, 0.0, 1.0])
    rays = build_rays(width, height, cam_pos, look_at, up, fov_deg)

    colors = _render_kernel(
        rays,
        float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]),
        starfield, disk_on, float(r_in), float(r_out), float(dt_base),
    )
    return colors.reshape(height, width, 3)