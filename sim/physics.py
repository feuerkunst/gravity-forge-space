import math
import numpy as np
from config import G, SOFTENING, TRAIL_MAX

_rng = np.random.default_rng()
_buffers = {}


def _get_buffers(n):
    buf = _buffers.get(n)
    if buf is None:
        buf = (
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
            np.empty(n, dtype=np.float32),
        )
        _buffers.clear()
        _buffers[n] = buf
    return buf


def step_physics(pos, vel, bodies, dt, jets_enabled=True):
    dt_f = np.float32(dt)
    eps2 = np.float32(SOFTENING * SOFTENING)
    nb = len(bodies)

    b_acc = [np.zeros(3, dtype=np.float32) for _ in range(nb)]
    for i in range(nb):
        if bodies[i]["kind"] == "repulsor":
            continue
        pi = bodies[i]["pos"]
        for j in range(nb):
            if i == j or bodies[j]["kind"] == "repulsor":
                continue
            d = bodies[j]["pos"] - pi
            dist2 = float(d @ d) + float(eps2) * 2.2
            b_acc[i] += d * (G * bodies[j]["mass"] / (dist2 * math.sqrt(dist2)))

    for i in range(nb):
        b = bodies[i]
        b["spin"] += b["spin_speed"] * dt
        if b["kind"] != "repulsor":
            b["vel"] += b_acc[i] * dt
            b["pos"] += b["vel"] * dt

        if b["kind"] not in ("core", "repulsor"):
            b["trail_acc"] += dt
            if b["trail_acc"] >= 0.045:
                b["trail_acc"] = 0.0
                if not b["trail"] or float(np.linalg.norm(b["pos"] - b["trail"][-1])) > 0.16:
                    b["trail"].append(b["pos"].copy())
                    if len(b["trail"]) > TRAIL_MAX:
                        b["trail"].pop(0)

    removed = set()
    for i in range(nb):
        if i in removed or bodies[i]["kind"] in ("core", "repulsor"):
            continue
        for j in range(i + 1, nb):
            if j in removed or bodies[j]["kind"] in ("core", "repulsor"):
                continue
            d = bodies[j]["pos"] - bodies[i]["pos"]
            dist = float(np.linalg.norm(d))
            merge_r = (bodies[i]["radius"] + bodies[j]["radius"]) * 0.72
            if dist < merge_r:
                if abs(bodies[i]["mass"]) >= abs(bodies[j]["mass"]):
                    big, small, idx_rem = bodies[i], bodies[j], j
                else:
                    big, small, idx_rem = bodies[j], bodies[i], i
                tot_m = big["mass"] + small["mass"]
                if tot_m > 0.1:
                    big["vel"] = (big["vel"] * big["mass"] + small["vel"] * small["mass"]) / tot_m
                    big["mass"] = tot_m
                    big["radius"] = min(1.85, (big["radius"] ** 3 + small["radius"] ** 3 * 0.4) ** (1.0 / 3.0))
                removed.add(idx_rem)
                if idx_rem == i:
                    break

    if removed:
        bodies[:] = [b for idx, b in enumerate(bodies) if idx not in removed]

    n = pos.shape[0]
    ax, ay, az, dx, dy, dz, dist2, tmp = _get_buffers(n)
    ax.fill(0.0)
    ay.fill(0.0)
    az.fill(0.0)

    px = pos[:, 0]
    py = pos[:, 1]
    pz = pos[:, 2]

    for b in bodies:
        bp = b["pos"]
        np.subtract(bp[0], px, out=dx)
        np.subtract(bp[1], py, out=dy)
        np.subtract(bp[2], pz, out=dz)

        np.multiply(dx, dx, out=dist2)
        np.multiply(dy, dy, out=tmp)
        dist2 += tmp
        np.multiply(dz, dz, out=tmp)
        dist2 += tmp

        if jets_enabled and b["kind"] == "bh":
            hr = max(0.52, b["radius"] * 0.85)
            caught = np.flatnonzero(dist2 < np.float32(hr * hr))
            if caught.size > 0:
                b["absorbed"] += int(caught.size)
                signs = _rng.choice([-1.0, 1.0], size=caught.size).astype(np.float32)[:, None]
                axis = b["jet_axis"][None, :]
                jitter = _rng.normal(0.0, 0.14, size=(caught.size, 3)).astype(np.float32)
                jdir = axis * signs + jitter
                jn = np.linalg.norm(jdir, axis=1, keepdims=True) + 1e-6
                jdir /= jn
                spd = _rng.uniform(11.5, 22.0, size=(caught.size, 1)).astype(np.float32)
                pos[caught] = bp[None, :] + jdir * (hr * 1.35)
                vel[caught] = b["vel"][None, :] + jdir * spd

        dist2 += eps2
        np.sqrt(dist2, out=tmp)
        tmp *= dist2
        np.divide(np.float32(G * b["mass"]), tmp, out=tmp)

        dx *= tmp
        ax += dx
        dy *= tmp
        ay += dy
        dz *= tmp
        az += dz

    ax *= dt_f
    ay *= dt_f
    az *= dt_f
    vel[:, 0] += ax
    vel[:, 1] += ay
    vel[:, 2] += az

    np.multiply(vel[:, 0], dt_f, out=dx)
    np.multiply(vel[:, 1], dt_f, out=dy)
    np.multiply(vel[:, 2], dt_f, out=dz)
    px += dx
    py += dy
    pz += dz
