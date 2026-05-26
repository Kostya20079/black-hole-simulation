import numpy as np
from numba import njit, prange
from PIL import Image
import time

from physics import rk4_step_nb

# ==== Constants ====
RS = 1.0
DT_BASE = 0.5
MAX_STEPS = 3000
ESCAPE_R = 60.0
DISK_R_IN = 3.0  * RS
DISK_R_OUT = 12.0 * RS

# Camera
CAM_POS_ARR = np.array([-20.0, 0.0, 5.0])
LOOK_AT_ARR = np.array([0.0, 0.0, 0.0])
UP_ARR = np.array([0.0, 0.0, 1.0])
FOV_DEG = 100.0

WIDTH = 1280
HEIGHT = 720

def generate_starfield(
    seed: int = 42, n_bright: int = 3000,
    width: int = 1024, height: int = 512,
) -> np.ndarray:
    """
    Creates an equirectangular star-map texture (height, width, 3).

    Stars have physically-motivated colours based on temperature:
        Hot blue stars -> blue-white
        Sun-like stars -> yellow-white
        Cool red giants -> orange-red

    Also includes a faint "Milky Way" brightness band at the equator.
    """
    # Initialize a deterministic random number generator and an empty RGB texture
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.float32)

    # ==== Dim background stars (noise-based) ====
    # Generate random noise for every pixel of the map
    noise = rng.random((height, width))
    # Pixels with a noise value > 0.997 become faint background stars (gray color)
    img[noise > 0.997] = 0.4

    # ---- Bright named stars ----
    # Randomize (X, Y) positions and base brightness for n_bright main stars
    sx = (rng.random(n_bright) * width ).astype(int)
    sy = (rng.random(n_bright) * height).astype(int)
    brightness = rng.random(n_bright) * 0.8 + 0.2

    # Star temperature parameter: 0 = cool/red, 1 = hot/blue
    temp = rng.random(n_bright)
    # Empirical RGB channel mixing model based on star temperature
    colors = np.stack([
        np.clip(1.5 - temp, 0, 1), # R channel: dominates at low temperatures
        np.clip(0.8 + 0.4*temp - (temp - 0.5)**2, 0, 1), # G channel: stable middle of the scale
        np.clip(temp * 1.5, 0, 1), # B channel: dominates at high temperatures
    ], axis=1)

    # Render each bright star with a glow effect (3x3 blur kernel)
    for i in range(n_bright):
        b, c = brightness[i], colors[i]
        # Loop surrounding the star center within a 1-pixel radius
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                # Apply modulo operations so the texture wraps around the edges (spherical mapping)
                py = (sy[i] + dy) % height
                px = (sx[i] + dx) % width
                # Star center has a weight of 1.0, surrounding glow has a weight of 0.22
                w  = 1.0 if (dy == 0 and dx == 0) else 0.22
                # Accumulate light energy into the texture while capping at absolute white (1.0)
                img[py, px] = np.minimum(img[py, px] + b * w * c, 1.0)

    # ---- Milky Way band ----
    # Create a vertical gradient to simulate the Milky Way galactic plane
    ys = np.arange(height)
    # Gaussian distribution centering the bright band at the map equator (mid-height)
    band = np.exp(-((ys - height * 0.5)**2) / (height * 0.06)**2)
    # Superimpose random dust noise modulated by the equatorial density band
    mw_noise = rng.random((height, width)) * band[:, None] * 0.09
    # Add the Milky Way glow to all three color channels simultaneously
    img += mw_noise[:, :, None]

    # Final clip to ensure values stay within the valid color range [0, 1]
    return np.clip(img, 0, 1)


@njit(cache=True)
def sample_sky_nb(vx, vy, vz, starfield):
    """Equirectangular starfield lookup — scalar direction components."""
    H = starfield.shape[0]
    W = starfield.shape[1]
    # Normalize the photon escape direction vector to a unit sphere
    r = (vx*vx + vy*vy + vz*vz) ** 0.5
    dx, dy, dz = vx/r, vy/r, vz/r

    # Convert Cartesian vector (X, Y, Z) to spherical coordinates (phi, theta)
    phi   = np.arctan2(dy, dx) # Azimuth angle in the XY plane [-pi, pi]
    sin_z = min(1.0, max(-1.0, dz)) # Protection against floating-point edge cases for arcsin
    theta = np.arcsin(sin_z) # Elevation angle (latitude) [-pi/2, pi/2]

    # Map spherical angles to normalized texture coordinates [0.0, 1.0]
    u = (phi / (2.0 * 3.141592653589793) + 0.5) % 1.0
    v = theta / 3.141592653589793 + 0.5

    # Map normalized coordinates to actual pixel indices (integers)
    xi = int(u * W) % W
    yi = int(v * H) % H
    # Return the RGB components of the sampled star/background from the sky map
    return starfield[yi, xi, 0], starfield[yi, xi, 1], starfield[yi, xi, 2]


@njit(cache=True)
def disk_color_nb(r, phi, doppler):
    """Disk colour model"""
    # Normalize the ray position inside the disk (0 = inner edge, 1 = outer edge)
    t = (r - DISK_R_IN) / (DISK_R_OUT - DISK_R_IN)
    t = max(0.0, min(1.0, t))

    # Base brightness profile: brightest at the center, dropping sharply towards the edges
    base = (1.0 - t) ** 1.8 * 0.88 + 0.12
    # Generate spiral arm structures/turbulences using sine functions
    turb = 0.15 * np.sin(7.0*phi + 4.0*t) * np.sin(2.5*t*3.14159)
    # Combine base brightness with turbulences and apply the relativistic Doppler effect
    bright = max(0.0, min(1.0, base + turb)) * max(0.05, min(6.0, doppler))
    bright = max(0.0, min(1.0, bright))

    # Model the disk's color temperature depending on the distance (t) from the black hole:
    if t < 0.15:
        # Inner region (extremely hot): blue-white light
        rc, gc, bc = 0.85, 0.80 + 0.15*(1 - t/0.15), 1.00
    elif t < 0.35:
        # Mid-inner region: transitioning into a hot, bright yellow
        tt = (t - 0.15) / 0.20
        rc, gc, bc = 1.00, 0.90 - 0.25*tt, 0.55 - 0.45*tt
    elif t < 0.60:
        # Mid-outer region: cooling matter transitioning to orange
        tt = (t - 0.35) / 0.25
        rc, gc, bc = 1.00 - 0.1*tt, 0.55 - 0.25*tt, 0.10 - 0.08*tt
    else:
        # Outer region (coolest): dim, fading red
        tt = (t - 0.60) / 0.40
        rc, gc, bc = 0.85 - 0.4*tt, 0.25 - 0.2*tt, 0.02

    # Return the final RGB color multiplied by the local brightness factor
    return (max(0.0, min(1.0, rc*bright)),
            max(0.0, min(1.0, gc*bright)),
            max(0.0, min(1.0, bc*bright)))


# ==== Parallel render kernel ====
@njit(parallel=True, cache=True)
def render_parallel(
    rays_flat: np.ndarray, # (H*W, 3) pre-flattened ray directions
    cam_px: float, cam_py: float, cam_pz: float, # camera position
    starfield: np.ndarray, # (SH, SW, 3)
    cam_dir_x: float, cam_dir_y: float, # camera direction in xy (for Doppler)
) -> np.ndarray:
    """
    Parallel pixel loop.

    prange(n) distributes iterations across all available CPU cores.
    Each pixel is independent so there are no race conditions.

    Returns (H*W, 3) float32 colour array.
    """
    n  = rays_flat.shape[0]
    out = np.zeros((n, 3), dtype=np.float32)

    # Multi-threaded loop processing each ray (pixel) independently
    for idx in prange(n):
        # Initialize the ray's starting position (camera position)
        px, py, pz = cam_px, cam_py, cam_pz
        # Fetch the initial velocity (direction) vector for this specific pixel
        vx = rays_flat[idx, 0]
        vy = rays_flat[idx, 1]
        vz = rays_flat[idx, 2]

        result_r = 0.0
        result_g = 0.0
        result_b = 0.0
        outcome = 0 # Final state flag: 0=ray escaped, 1=captured by singularity, 2=hit the disk

        # Main numerical integration loop (Ray Marching) for the light trajectory
        for _ in range(MAX_STEPS):
            # Calculate the current Euclidean distance from the center of the black hole
            r = (px*px + py*py + pz*pz) ** 0.5

            # Event horizon test: ray crossed the point of no return (includes a safety margin)
            if r < RS * 0.95:
                outcome = 1;  break

            # Escape test: ray is far enough that gravity no longer significantly bends it
            if r > ESCAPE_R:
                outcome = 0;  break

            # Adaptive Step Size:
            # The closer to the black hole, the smaller the step (dt) to accurately calculate strong bending.
            closeness = max(r / RS - 1.0, 0.05)
            dt = DT_BASE * min(closeness * 0.5, 1.0)
            dt = max(dt, DT_BASE * 0.01) # Lower bound to prevent the step from reaching zero

            # Perform a step of the Runge-Kutta 4th order method to determine new position and velocity
            new_px, new_py, new_pz, new_vx, new_vy, new_vz = rk4_step_nb(px, py, pz, vx, vy, vz, dt)

            # ---- Disk crossing check ----
            # Check if the ray crossed the equatorial plane (change of sign in the Z coordinate)
            z0, z1 = pz, new_pz
            if z0 * z1 <= 0.0 and abs(z0 - z1) > 1e-10:
                # Linear interpolation of the intersection moment with the Z=0 plane (fractional step t_c)
                t_c = -z0 / (z1 - z0)
                # Calculate exact (X, Y) coordinates of the hit on the disk plane
                hx = px + t_c*(new_px - px)
                hy = py + t_c*(new_py - py)
                r_hit = (hx*hx + hy*hy) ** 0.5

                # Check if the intersection point lies within the geometric boundaries of the accretion disk
                if DISK_R_IN <= r_hit <= DISK_R_OUT:
                    phi_hit = np.arctan2(hy, hx)

                    # --- Relativistic Doppler Effect Calculation ---
                    # Determine the orbital velocity of matter at this radius (Keplerian approximation)
                    v_orb = min((0.5 / max(r_hit, 0.1)) ** 0.5, 0.45)
                    r_hat_x = hx / r_hit;  r_hat_y = hy / r_hit

                    # Disk matter velocity vector (perpendicular to the radius vector - circular motion)
                    vdx = -r_hat_y * v_orb;
                    vdy = r_hat_x * v_orb

                    # Dot product of disk velocity and observation direction (relative velocity)
                    beta = vdx*cam_dir_x + vdy*cam_dir_y
                    beta = max(-0.90, min(0.90, beta)) # Safeguard against asymptotes near speed of light 'c'
                    # Doppler factor cubed to account for relativistic aberration and beaming
                    D3 = ((1.0 + beta) / (1.0 - beta)) ** 1.5

                    # Fetch the color based on position and the Doppler shift factor
                    rc, gc, bc = disk_color_nb(r_hit, phi_hit, D3)

                    result_r, result_g, result_b = rc, gc, bc

                    outcome = 2
                    break # Ray hit opaque matter - terminate further tracking for this ray

            # Update particle state to the new position before the next physical iteration loop
            px, py, pz = new_px, new_py, new_pz
            vx, vy, vz = new_vx, new_vy, new_vz

        # Assign colors based on the final outcome ("fate") of the light ray
        if outcome == 0:
            # Ray escaped: sample color from the cosmic background starfield
            result_r, result_g, result_b = sample_sky_nb(vx, vy, vz, starfield)
        elif outcome == 1:
            # Ray fell into the black hole: absorbed (pure black pixel)
            result_r = result_g = result_b = 0.0

        # Store the final pixel color into the flattened output array
        out[idx, 0] = result_r
        out[idx, 1] = result_g
        out[idx, 2] = result_b

    return out


# ==== Camera ray builder ====
def build_rays_flat(width, height, cam_pos, look_at, up, fov_deg):
    """Returns (H*W, 3) flattened ray array for the parallel kernel."""
    # Compute the orthonormal basis (coordinate system) of the 3D camera:
    # Forward vector
    fwd = look_at - cam_pos;
    fwd /= np.linalg.norm(fwd)
    # Right vector generated via cross product
    rgt = np.cross(fwd, up);
    rgt /= np.linalg.norm(rgt)
    # Corrected Up vector ensuring strict axis orthogonality
    cup = np.cross(rgt, fwd)

    # Convert Field of View (FOV) angle to physical viewing window size (frustum)
    hw = np.tan(np.radians(fov_deg / 2.0))
    hh = hw / (width / height) # Scale height based on Aspect Ratio

    # Generate normalized pixel positions on the screen grid from -1 to 1
    u = ((np.arange(width)  + 0.5) / width  * 2.0 - 1.0) * hw
    v = ((np.arange(height) + 0.5) / height * 2.0 - 1.0) * hh
    # Invert V axis (v[::-1]) so the top of the image corresponds to positive values in 3D space
    uu, vv = np.meshgrid(u, v[::-1])

    # Linear combination of camera basis vectors to determine the 3D direction for each pixel ray
    dirs = fwd + uu[..., None]*rgt + vv[..., None]*cup
    # Normalize direction vectors, converting them to initial unit velocity vectors of light
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    # Flatten the structure from a 2D image to a 1D list of vectors (preparation for multi-threading)
    return dirs.reshape(-1, 3).astype(np.float64)

def render_hd():
    print("=" * 60)
    print("Phase Numba-Accelerated HD Render")
    print(f"Resolution   : {WIDTH} × {HEIGHT}  ({WIDTH*HEIGHT:,} pixels)")
    print(f"CPU cores    : {nb.get_num_threads()}")
    print("=" * 60)

    print("\n[1/4] Generating star background (2048×1024)...")
    starfield = generate_starfield(
        n_bright=6000, width=2048, height=1024
    ).astype(np.float32)

    print("[2/4] Building camera rays...")
    rays_flat = build_rays_flat(
        WIDTH, HEIGHT, CAM_POS_ARR, LOOK_AT_ARR, UP_ARR, FOV_DEG
    )

    # Camera direction (xy projection, used for Doppler)
    fwd_xy = (LOOK_AT_ARR - CAM_POS_ARR)[:2]
    fwd_xy /= np.linalg.norm(fwd_xy)
    cam_dir_x = float(fwd_xy[0])
    cam_dir_y = float(fwd_xy[1])

    print("[3/4] Compiling Numba kernels (first run only)...")
    t_compile = time.time()
    # Warm-up with a single pixel to trigger compilation
    dummy = render_parallel(
        rays_flat[:4], *CAM_POS_ARR, starfield, cam_dir_x, cam_dir_y
    )
    print(f"      Compiled in {time.time()-t_compile:.1f}s")

    print(f"[4/4] Rendering {WIDTH*HEIGHT:,} pixels on {nb.get_num_threads()} cores...")
    t0 = time.time()

    colors = render_parallel(
        rays_flat,
        float(CAM_POS_ARR[0]), float(CAM_POS_ARR[1]), float(CAM_POS_ARR[2]),
        starfield, cam_dir_x, cam_dir_y,
    )

    elapsed = time.time() - t0
    pps = WIDTH * HEIGHT / elapsed
    print(f"\nRender time : {elapsed:.1f}s  ({pps:,.0f} pixels/s)")

    image = colors.reshape(HEIGHT, WIDTH, 3)
    out = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(out).save("./plots/hd_black_hole.png")
    print("Saved-> hd_black_hole.png")


if __name__ == "__main__":
    render_hd()