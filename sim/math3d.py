import math
import numpy as np


def rot_mat(rx, ry, rz):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return mz @ my @ mx


def normalize(v):
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return (v / n).astype(np.float32)
