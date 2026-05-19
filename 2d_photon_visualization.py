import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from physics import rk4_step, RS

def trace_photon_2d(
        start_position_2d: list,
        start_direction_2d: list,
        dt: float = 0.05, # time step
        max_dist: float = 50.0, # photon max distance range
        max_steps: int = 6000
) -> tuple:
    """
    Traces one photon through the equatorial plane (z = 0).

    The 2D (x,y) vectors are embedded in 3D space with z=0 so the
    shared physics engine can be used without modification.

    Adaptive step size:
        Far from Black Hole -> large dt (photon barely curves, each step cheap)
        Near Black Hole -> tiny dt (high curvature, need precision)

    Returns:
        trajectory : (N, 2) array of (x, y) positions
        status : 'captured' | 'escaped'
    """

    # converting basic state and converting from 2D to 3D array
    state = np.array([
        start_position_2d[0], start_position_2d[1], 0.0, # using 0 for 2D visualization
        start_direction_2d[0], start_direction_2d[1], 0.0,
    ], dtype=float)

    trajectory = [state[:2].copy()]
    status = "escaped"

    for _ in range(max_steps):
        r = np.linalg.norm(state[:3]) # how far is photon

        # if photon is inside event horizon
        if r < RS:
            status = "captured"
            break

        # if photon went very far away
        if r > max_dist:
            status = "escaped"
            break

        # measures how close photon is to event horizon
        closeness = max(r / RS - 1.0, 0.5)

        # Near Black Hole: tiny timestep, high precision
        # Far away from Black Hole: larger timestep, faster simulation
        adaptive_dt = dt * min(closeness* 0.5, 1.0)
        adaptive_dt = max(adaptive_dt, dt * 0.005) # prevents timestep from becoming too tiny

        state = rk4_step(state, adaptive_dt)
        trajectory.append(state[:2].copy()) # add new trajectory

    return np.array(trajectory), status


def visualize():
    """Renders and saves the 2D trajectory plot."""

    fig, ax = plt.subplots(figsize=(14, 10), facecolor="#080814")
    ax.set_facecolor("#080814")
    ax.set_aspect("equal")

    # filled black-hole interior
    ax.add_patch(patches.Circle((0, 0), RS, color="#0a0a12", zorder=10))
    # event horizon ring
    ax.add_patch(patches.Circle(
        (0, 0), RS, color="white", fill=False, lw=2, zorder=11,
        label=f"Event Horizon  r = Rs = {RS:.0f}",
    ))
    # photon sphere (unstable circular orbit)
    ax.add_patch(patches.Circle(
        (0, 0), 1.5 * RS, color="#ffcc44", fill=False,
        lw=1, ls="--", alpha=0.5, zorder=9,
        label=f"Photon Sphere  r = 1.5 Rs = {1.5*RS:.0f}",
    ))
    # innermost stable circular orbit
    ax.add_patch(patches.Circle(
        (0, 0), 3 * RS, color="#ff7733", fill=False,
        lw=0.8, ls=":", alpha=0.35, zorder=9,
        label=f"Innermost Stable Circular Orbit  r = 3 Rs = {3*RS:.0f}",
    ))

    # Impact parameter b = perpendicular distance from Black Hole centre to the incoming ray
    b_critical = 1.5 * np.sqrt(3) * RS
    impacts = np.linspace(-10, 10, 60) # generating races
    cmap = plt.cm.plasma

    for b in impacts:
        t = abs(b) / 10.0
        color = cmap(t)

        traj, status = trace_photon_2d(
            start_position_2d=[-22.0, b],
            start_direction_2d=[1.0, 0.0],
        )

        # check if ray close to photon-sphere threshold
        near_critical = abs(abs(b) - b_critical) < 0.25
        lw = 2.0 if near_critical else 0.6
        alpha = 1.0 if near_critical else 0.75 # transparency

        # plotting trajectories
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=lw, alpha=alpha, zorder=5)

    for sign in [+1, -1]:
        b_crit_ray = sign * (b_critical + 0.04)
        traj, _ = trace_photon_2d(
            start_position_2d=[-22.0, b_crit_ray],
            start_direction_2d=[1.0, 0.0],
            dt=0.02,
            max_dist=70,
            max_steps=15000, # more iterations
        )
        ax.plot(traj[:, 0], traj[:, 1], color="#ffee00", lw=2.2, zorder=8, alpha=0.9)

    # === Colourbar ===
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 10))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Impact parameter |b|  [GM/c²]",
                   color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # === Annotations ===
    ax.annotate(
        f"Critical ray\nb_crit_ray ≈ {b_critical:.2f}",
        xy=(3, 5.4), xytext=(8, 9),
        color="#ffee00", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="#ffee00", lw=1.2),
    )

    # === Labels / styling ===
    ax.set_xlim(-25, 25)
    ax.set_ylim(-18, 18)
    ax.set_xlabel("x  [GM/c²]", color="#aaaacc", fontsize=12)
    ax.set_ylabel("y  [GM/c²]", color="#aaaacc", fontsize=12)
    ax.set_title(
        "Photon Trajectories - Schwarzschild Black Hole\n"
        "Geometric units:  G = c = M = 1, Rs = 2",
        color="white", fontsize=14, pad=14,
    )
    ax.tick_params(colors="#aaaacc")
    for sp in ax.spines.values():
        sp.set_edgecolor("#222244")

    ax.legend(
        facecolor="#12121e", edgecolor="#333355",
        labelcolor="white", fontsize=9, loc="upper right",
    )

    plt.tight_layout()

    # saving plot
    plt.savefig("./plots/photon_visualization_2d.png", dpi=150, bbox_inches="tight", facecolor="#080814")
    print("Saved → photon_visualization_2d.png")

    plt.show()


if __name__ == '__main__':
    visualize()



