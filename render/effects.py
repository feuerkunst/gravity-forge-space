import numpy as np
from config import W, H, FOCAL


def box_blur(img, passes=2, buf_a=None, buf_b=None):
    if buf_a is None:
        buf_a = np.empty_like(img)
    if buf_b is None:
        buf_b = np.empty_like(img)

    src = img
    for _ in range(passes):
        np.multiply(src, 0.5, out=buf_a)
        buf_a[1:] += src[:-1] * 0.25
        buf_a[:-1] += src[1:] * 0.25

        np.multiply(buf_a, 0.5, out=buf_b)
        buf_b[:, 1:] += buf_a[:, :-1] * 0.25
        buf_b[:, :-1] += buf_a[:, 1:] * 0.25
        src = buf_b
    return src


def paint_background_stars(frame, bg_dirs, bg_cols, cam_basis):
    cam_d = bg_dirs @ cam_basis
    cz = cam_d[:, 2]
    vis = cz > 0.12
    if not np.any(vis):
        return
    dm = cam_d[vis]
    inv_z = FOCAL / dm[:, 2]
    sx = (dm[:, 0] * inv_z + W * 0.5).astype(np.int32)
    sy = (dm[:, 1] * inv_z + H * 0.5).astype(np.int32)
    ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if np.any(ok):
        frame[sx[ok], sy[ok]] = np.maximum(frame[sx[ok], sy[ok]], bg_cols[vis][ok])
