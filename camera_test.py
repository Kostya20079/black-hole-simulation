import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from physics import RS

CAMERA_POS = np.array([-20.0, 0.0, 0.0]) # x = -20
LOOK_AT = np.array([0,0,0])
UP_VECTOR = np.array([0,0,1]) # up vector

FOV_DEG = 150 # zoom

# resolution
WIDTH = 640
HEIGHT = 360

def build_camera_rays(
    width: int, height: int,
    cam_pos: np.ndarray, look_at: np.ndarray,
    up: np.ndarray, fov_deg: float,
) -> np.ndarray:
    forward = cam_pos - look_at
    forward = forward / np.linalg.norm(forward) # where camera looks

    right = np.cross(forward, up)
    right = right / np.linalg.norm(right) # horizontal screen direction

    cam_up = np.cross(right, forward) # vertical screen direction

    # screen extents
    aspect = width / height
    half_w = np.tan(np.radians(fov_deg / 2.0))
    half_h = half_w / aspect

    # compute all pixel centres at once
    ix = np.arange(width, dtype=float)
    iy = np.arange(height, dtype=float)

    # normalised device coordinates, pixel centres at +-0.5
    u_ndc = ((ix + 0.5) / width * 2.0 - 1.0) * half_w
    v_ndc = ((iy + 0.5) / height * 2.0 - 1.0) * half_h

    # 2D grids, shape (height, width)
    uu, vv = np.meshgrid(u_ndc, v_ndc[::-1])

    # dirs shape (height, width, 3)
    dirs = (
            forward[np.newaxis, np.newaxis, :]
            + uu[:, :, np.newaxis] * right[np.newaxis, np.newaxis, :]
            + vv[:, :, np.newaxis] * cam_up[np.newaxis, np.newaxis, :]
    )

    # normalise every direction vector
    norms = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = dirs / norms

    return dirs  # shape: (H, W, 3)


def ray_sphere_intersect(
        ray_origin: np.ndarray, ray_dirs: np.ndarray,
        sphere_center: np.ndarray, sphere_radius: float,
) -> np.ndarray:
    oc = ray_origin - sphere_center  # vector from sphere centre to eye

    # B -> dot product of oc with each direction
    B = 2.0 * np.einsum("ijk,k->ij", ray_dirs, oc)
    C = float(np.dot(oc, oc)) - sphere_radius ** 2

    discriminant = B ** 2 - 4.0 * C
    return discriminant >= 0.0


def render_camera_test():
    print(f"Camera geometry test  ({WIDTH}×{HEIGHT})")

    rays = build_camera_rays(WIDTH, HEIGHT, CAMERA_POS, LOOK_AT, UP_VECTOR, FOV_DEG)

    # Test 1: sphere hit (flat space, no gravity)
    hit_mask = ray_sphere_intersect(
        CAMERA_POS, rays,
        sphere_center=np.zeros(3),
        sphere_radius=RS,
    )

    flat_render = np.ones((HEIGHT, WIDTH, 3)) * 0.18  # dark grey background
    flat_render[hit_mask] = [0.0, 0.0, 0.0]  # black sphere

    # Test 2: direction colour map
    # Map each ray's xyz component to an RGB channel.
    dir_rgb = (rays + 1.0) / 2.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor="#111")

    axes[0].imshow(flat_render)
    axes[0].set_title(
        "Flat-space sphere test",
        color="white", fontsize=11,
    )
    axes[0].axis("off")

    axes[1].imshow(dir_rgb)
    axes[1].set_title(
        "Ray direction colour map\n"
        "R=x  G=y  B=z",
        color="white", fontsize=11,
    )
    axes[1].axis("off")

    # annotations on left panel
    axes[0].annotate(
        f"Event Horizon\nr = RS = {RS:.0f}", xy=(WIDTH // 2, HEIGHT // 2),
        xytext=(WIDTH // 2 + 80, HEIGHT // 2 - 80),
        color="white", fontsize=9,
        arrowprops=dict(arrowstyle="->", color="white"),
    )

    plt.suptitle(
        "Camera Geometry",
        color="white", fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig("./plots/camera_test1.png", dpi=150,
                bbox_inches="tight", facecolor="#111")
    print("Saved!")
    print(f"Pixels hitting sphere: {hit_mask.sum():,}  /  {WIDTH * HEIGHT:,}")
    plt.show()

    return rays

if __name__ == "__main__":
    render_camera_test()