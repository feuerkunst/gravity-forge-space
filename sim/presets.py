import math
import time
import numpy as np
from config import G, SOFTENING, BG_STAR_COUNT
from sim.math3d import rot_mat
from sim.bodies import make_body, make_core

PRESET_INFO = [
    (1, "Spiral Merger", "Two spiral galaxies on a tidal collision course"),
    (2, "Ring Collision", "Head-on impact creating an expanding starburst ring"),
    (3, "Three-Body Chaos", "Three unstable galaxies tearing into orbital filaments"),
    (4, "Black Hole Shredder", "A supermassive black hole shredding a passing spiral"),
    (5, "Planetary System", "Central star with orbiting planets inside a dust disk"),
]


def make_background_stars(seed=42):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1.0, 1.0, BG_STAR_COUNT)
    phi = rng.uniform(0.0, 2.0 * math.pi, BG_STAR_COUNT)
    s = np.sqrt(np.maximum(0.0, 1.0 - u * u))
    dirs = np.stack([s * np.cos(phi), u, s * np.sin(phi)], axis=-1).astype(np.float32)

    temp = rng.random(BG_STAR_COUNT)
    cols = np.empty((BG_STAR_COUNT, 3), dtype=np.float32)
    for i in range(BG_STAR_COUNT):
        t = temp[i]
        if t < 0.18:
            c = (0.68, 0.80, 1.00)
        elif t < 0.62:
            c = (0.94, 0.95, 1.00)
        elif t < 0.86:
            c = (1.00, 0.91, 0.74)
        else:
            c = (1.00, 0.72, 0.52)
        b = 0.20 + 0.80 * (rng.random() ** 2.6)
        cols[i] = (c[0] * b, c[1] * b, c[2] * b)
    return dirs, (np.clip(cols, 0.0, 1.0) * 215.0).astype(np.uint8)


def make_galaxy(rng, count, center, vel, core_mass, radius, arms, tilt, spin, c_inner, c_outer):
    n_bulge = int(count * 0.14)
    n_disk = count - n_bulge

    r_d = 0.55 + (radius - 0.55) * (rng.random(n_disk) ** 1.42)
    arm_id = rng.integers(0, arms, n_disk)
    twist = 3.85 * (r_d / radius) * spin
    spread = 0.30 + 0.52 * (r_d / radius)
    ang_d = arm_id * (2.0 * math.pi / arms) + twist + rng.normal(0.0, spread, n_disk)
    z_d = rng.normal(0.0, 0.38 * np.exp(-r_d / (radius * 0.45)) + 0.07, n_disk)

    r_b = 0.12 + (radius * 0.24) * (rng.random(n_bulge) ** 1.8)
    ang_b = rng.uniform(0.0, 2.0 * math.pi, n_bulge)
    z_b = rng.normal(0.0, 0.42 * (1.0 - r_b / (radius * 0.35)), n_bulge)

    r = np.concatenate([r_b, r_d])
    ang = np.concatenate([ang_b, ang_d])
    z = np.concatenate([z_b, z_d])

    p_loc = np.stack([r * np.cos(ang), r * np.sin(ang), z], axis=-1).astype(np.float32)

    r2 = r * r + SOFTENING * SOFTENING
    speed = r * np.sqrt((G * core_mass) / (r2 * np.sqrt(r2)))
    speed *= rng.uniform(0.95, 1.05, count)
    vx = -np.sin(ang) * speed * spin
    vy = np.cos(ang) * speed * spin
    vz = rng.normal(0.0, 0.11, count)
    v_loc = np.stack([vx, vy, vz], axis=-1).astype(np.float32)

    m = rot_mat(*tilt)
    pos = p_loc @ m.T + np.array(center, dtype=np.float32)
    v_out = v_loc @ m.T + np.array(vel, dtype=np.float32)

    t = np.clip((r / radius) ** 0.72, 0.0, 1.0)[:, None]
    ci = np.array(c_inner, dtype=np.float32)
    co = np.array(c_outer, dtype=np.float32)
    cols = ci * (1.0 - t) + co * t

    hot_knots = rng.random(count) < 0.08
    cols[hot_knots] = cols[hot_knots] * 0.45 + np.array([0.45, 0.72, 1.0], dtype=np.float32) * 0.55
    red_giants = rng.random(count) < 0.06
    cols[red_giants] = cols[red_giants] * 0.45 + np.array([1.0, 0.55, 0.30], dtype=np.float32) * 0.55

    cols *= rng.uniform(0.72, 1.24, (count, 1)).astype(np.float32)
    return pos, v_out, cols


def make_planet_system_disk(rng, count, star_mass):
    r = 2.6 + 17.5 * (rng.random(count) ** 1.15)
    ang = rng.uniform(0.0, 2.0 * math.pi, count)
    z = rng.normal(0.0, 0.16 + 0.03 * r, count)
    pos = np.stack([r * np.cos(ang), z, r * np.sin(ang)], axis=-1).astype(np.float32)

    r2 = r * r + SOFTENING * SOFTENING
    speed = r * np.sqrt((G * star_mass) / (r2 * np.sqrt(r2))) * rng.uniform(0.97, 1.03, count)
    vel = np.stack([-np.sin(ang) * speed, rng.normal(0.0, 0.04, count), np.cos(ang) * speed], axis=-1).astype(np.float32)

    t = np.clip((r - 2.6) / 17.5, 0.0, 1.0)[:, None]
    ci = np.array([1.0, 0.84, 0.58], dtype=np.float32)
    co = np.array([0.28, 0.56, 0.96], dtype=np.float32)
    cols = (ci * (1.0 - t) + co * t) * rng.uniform(0.45, 1.05, (count, 1)).astype(np.float32)
    return pos, vel, cols


def load_preset(preset_id, n_stars):
    rng = np.random.default_rng(int(time.time() * 100) % 100000)
    parts_p, parts_v, parts_c = [], [], []
    bodies = []

    if preset_id == 5:
        name = "Planetary System"
        star_mass = 36.0
        bodies.append(make_body("star", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), mass=star_mass))
        pos, vel, cols = make_planet_system_disk(rng, n_stars, star_mass)

        orbits = [
            ("lava", 3.6, 0.2),
            ("earth", 6.4, 1.7),
            ("rocky", 9.0, 3.1),
            ("gas", 12.8, 4.6),
            ("ice", 17.2, 0.9),
        ]
        for pid, rad, phi in orbits:
            p = np.array([rad * math.cos(phi), 0.0, rad * math.sin(phi)], dtype=np.float32)
            r2 = rad * rad + SOFTENING * SOFTENING
            spd = rad * math.sqrt((G * star_mass) / (r2 * math.sqrt(r2)))
            v = np.array([-math.sin(phi) * spd, 0.0, math.cos(phi) * spd], dtype=np.float32)
            tilt = (0.42, 0.15, 0.2) if pid in ("gas", "ice") else (0.18, 0.1, 0.0)
            bodies.append(make_body(pid, p, v, tilt=tilt))

        return pos, vel, cols, bodies, name

    if preset_id == 1:
        name = "Spiral Merger"
        specs = [
            (n_stars // 2, (-12.5, -2.5, 0.0), (1.35, 1.15, 0.15), 26.0, 10.5, 2, (0.55, 0.25, 0.1), 1.0, (1.0, 0.92, 0.75), (0.18, 0.62, 1.0)),
            (n_stars - n_stars // 2, (12.5, 2.5, 0.0), (-1.35, -1.15, -0.15), 24.0, 9.8, 3, (-0.75, 0.45, -0.2), -1.0, (1.0, 0.90, 0.70), (1.0, 0.28, 0.52)),
        ]
    elif preset_id == 2:
        name = "Ring Collision"
        n1 = int(n_stars * 0.68)
        specs = [
            (n1, (0.0, 0.0, -5.0), (0.0, 0.0, 0.55), 32.0, 12.5, 2, (0.05, 0.05, 0.0), 1.0, (1.0, 0.95, 0.85), (0.12, 0.85, 0.95)),
            (n_stars - n1, (0.4, -0.2, 15.0), (0.0, 0.0, -3.4), 19.0, 6.2, 2, (1.1, 0.3, 0.0), -1.0, (1.0, 0.86, 0.55), (1.0, 0.42, 0.12)),
        ]
    elif preset_id == 3:
        name = "Three-Body Chaos"
        n1 = n_stars // 3
        n2 = n_stars // 3
        n3 = n_stars - n1 - n2
        specs = [
            (n1, (-13.0, -7.0, 2.0), (1.6, 1.2, -0.3), 22.0, 8.5, 2, (0.4, -0.5, 0.2), 1.0, (1.0, 0.92, 0.78), (0.15, 0.72, 1.0)),
            (n2, (13.0, -6.0, -2.0), (-1.4, 1.5, 0.4), 21.0, 8.2, 2, (-0.6, 0.3, 0.5), -1.0, (1.0, 0.88, 0.72), (1.0, 0.25, 0.62)),
            (n3, (0.0, 14.0, 0.0), (-0.2, -2.6, -0.1), 20.0, 8.0, 3, (0.9, 0.8, -0.3), 1.0, (0.95, 1.0, 0.85), (0.28, 1.0, 0.65)),
        ]
    else:
        name = "Black Hole Shredder"
        specs = [
            (n_stars, (-11.0, -3.5, 0.0), (1.9, 1.65, 0.0), 22.0, 10.5, 2, (0.35, -0.25, 0.0), 1.0, (1.0, 0.95, 0.82), (0.32, 0.55, 1.0)),
        ]
        bodies.append(make_body("bh", (4.5, 1.5, 0.0), (-0.4, -0.3, 0.0), mass=62.0, tilt=(0.55, 0.25, 0.15)))

    for sp in specs:
        p, v, c = make_galaxy(rng, *sp)
        parts_p.append(p)
        parts_v.append(v)
        parts_c.append(c)
        bodies.append(make_core(sp[1], sp[2], sp[3]))

    pos = np.vstack(parts_p)
    vel = np.vstack(parts_v)
    cols = np.vstack(parts_c)
    return pos, vel, cols, bodies, name
