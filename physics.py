import numpy as np

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


def rk4_step(state: np.ndarray, dt: float) -> np.ndarray:
    """
        Advances photon state by one time-step using 4th-order Runge-Kutta.

        Args:
            state: photon state vector [pos, vel]
            dt: integration step siz

        Returns:
            new_state: updated [pos, vel] after dt
        """

    k1 = derivatives(state)
    k2 = derivatives(state + 0.5 * dt * k1)
    k3 = derivatives(state + 0.5 * dt * k2)
    k4 = derivatives(state + dt * k3)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)