import numpy as np
from numba import njit

"""
Core geodesic engine

UNIT SYSTEM — Geometric units:
G = c = M = 1
Schwarzschild radius: Rs = 2·G·M/c^2 = 2
Speed of light: c = 1
Black hole mass: M = 1

This means all distances are measured in units of GM/c^2
and times in GM/c^3. The event horizon is a sphere of radius 2
"""

# ===== CONSTANTS =====
M = 1.0 # Mass of Black Hole
RS = 2 * M # Schwarzschild radius


def derivatives(state: np.ndarray) -> np.ndarray:
    """
    Computes [velocity, acceleration] for a photon in Schwarzschild spacetime.

    Args:
        state: np.array([x, y, z, vx, vy, vz])

    Returns:
        np.array([vx, vy, vz, ax, ay, az])
    """

    pos = state[:3]
    vel = state[3:]

    r = np.linalg.norm(pos)

    # Angular-momentum vector
    L = np.cross(pos, vel)
    L2 = np.dot(L, L)

    # Geodesic acceleration directed toward center
    # magnitude grows as 1/r^4 so it becomes enormous near the horizon
    accel = -(1.5 *  RS * L2 / r**5) * pos

    return np.concatenate([vel, accel])

@njit(cache=True)
def geodesic_accel(px, py, pz, vx, vy, vz):
    """
    Computes the geodesic acceleration components for a photon.
    Returns (ax, ay, az).
    """
    # Cross product h = pos × vel
    hx = py*vz - pz*vy
    hy = pz*vx - px*vz
    hz = px*vy - py*vx

    h2 = hx*hx + hy*hy + hz*hz # |h|^2
    r  = (px*px + py*py + pz*pz) ** 0.5 # |pos|
    r5 = r * r * r * r * r

    factor = -1.5 * RS * h2 / r5
    return factor*px, factor*py, factor*pz


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """
        Advances photon state by one time-step using 4th-order Runge-Kutta.

        Args:
            state: photon state vector [pos, vel]
            dt: integration step size

        Returns:
            new_state: updated [pos, vel] after dt
    """

    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

@njit(cache=True)
def rk4_step_nb(px, py, pz, vx, vy, vz, dt):
    """
    One RK4 step. Input and output are scalar components.
    Returns (new_px, new_py, new_pz, new_vx, new_vy, new_vz).
    """
    # k1
    k1vx, k1vy, k1vz = vx, vy, vz
    k1ax, k1ay, k1az = geodesic_accel(px, py, pz, vx, vy, vz)

    # k2
    p2x = px + 0.5*dt*k1vx;  p2y = py + 0.5*dt*k1vy;  p2z = pz + 0.5*dt*k1vz
    v2x = vx + 0.5*dt*k1ax;  v2y = vy + 0.5*dt*k1ay;  v2z = vz + 0.5*dt*k1az
    k2vx, k2vy, k2vz = v2x, v2y, v2z
    k2ax, k2ay, k2az = geodesic_accel(p2x, p2y, p2z, v2x, v2y, v2z)

    # k3
    p3x = px + 0.5*dt*k2vx;  p3y = py + 0.5*dt*k2vy;  p3z = pz + 0.5*dt*k2vz
    v3x = vx + 0.5*dt*k2ax;  v3y = vy + 0.5*dt*k2ay;  v3z = vz + 0.5*dt*k2az
    k3vx, k3vy, k3vz = v3x, v3y, v3z
    k3ax, k3ay, k3az = geodesic_accel(p3x, p3y, p3z, v3x, v3y, v3z)

    # k4
    p4x = px + dt*k3vx;  p4y = py + dt*k3vy;  p4z = pz + dt*k3vz
    v4x = vx + dt*k3ax;  v4y = vy + dt*k3ay;  v4z = vz + dt*k3az
    k4ax, k4ay, k4az = geodesic_accel(p4x, p4y, p4z, v4x, v4y, v4z)

    inv6 = dt / 6.0
    new_px = px + inv6*(k1vx + 2*k2vx + 2*k3vx + (v4x))
    new_py = py + inv6*(k1vy + 2*k2vy + 2*k3vy + (v4y))
    new_pz = pz + inv6*(k1vz + 2*k2vz + 2*k3vz + (v4z))
    new_vx = vx + inv6*(k1ax + 2*k2ax + 2*k3ax + k4ax)
    new_vy = vy + inv6*(k1ay + 2*k2ay + 2*k3ay + k4ay)
    new_vz = vz + inv6*(k1az + 2*k2az + 2*k3az + k4az)

    return new_px, new_py, new_pz, new_vx, new_vy, new_vz

# Adaptive step size helper
@njit(cache=True, inline='always')
def adaptive_dt(r, dt_base):
    """
    Scale dt by proximity to horizon.

    Deep in the field (r ≈ Rs) -> tiny steps for accuracy.
    Far from the BH -> large steps for speed.

    Returns dt clamped to [dt_base*0.01, dt_base].
    """
    closeness = r / RS - 1.0
    if closeness < 0.05:
        closeness = 0.05
    dt = dt_base * (closeness * 0.5)
    if dt > dt_base:
        dt = dt_base
    if dt < dt_base*0.01:
        dt = dt_base * 0.01
    return dt