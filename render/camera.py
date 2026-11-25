import math
import numpy as np
from config import W, H, FOCAL


def get_camera(yaw, pitch, dist, target):
    cp = math.cos(pitch)
    cam_pos = target + np.array([
        dist * cp * math.sin(yaw),
        dist * math.sin(pitch),
        -dist * cp * math.cos(yaw),
    ], dtype=np.float32)

    fwd = target - cam_pos
    fwd /= max(1e-6, float(np.linalg.norm(fwd)))
    right = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), fwd)
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-5 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    up = np.cross(fwd, right)
    return cam_pos, fwd, right, up


def screen_to_world(mx, my, target, right, up, dist):
    xc = (mx - W * 0.5) * dist / FOCAL
    yc = -(my - H * 0.5) * dist / FOCAL
    return target + right * xc + up * yc


def world_to_screen(pt, cam_pos, fwd, right, up):
    d = pt - cam_pos
    z = float(d @ fwd)
    if z <= 0.6:
        return None
    inv_z = FOCAL / z
    sx = int((d @ right) * inv_z + W * 0.5)
    sy = int(-(d @ up) * inv_z + H * 0.5)
    return sx, sy, z


def pick_body(mx, my, bodies, cam_pos, fwd, right, up, scale_mul=1.75):
    best = None
    best_z = float("inf")

    for b in bodies:
        if b["kind"] == "core":
            continue
        proj = world_to_screen(b["pos"], cam_pos, fwd, right, up)
        if proj is None:
            continue
        sx, sy, z = proj
        rad_px = max(14, min(130, int(b["radius"] * scale_mul * (FOCAL / z)))) + 6
        dx = mx - sx
        dy = my - sy
        if dx * dx + dy * dy <= rad_px * rad_px and z < best_z:
            best_z = z
            best = b

    return best
