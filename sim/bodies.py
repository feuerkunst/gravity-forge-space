import math
import numpy as np
from config import G, SOFTENING
from sim.math3d import rot_mat, normalize

SPAWN_CATALOG = [
    {
        "id": "earth",
        "name": "Earth",
        "kind": "planet",
        "subtype": "earth",
        "mass": 3.2,
        "radius": 0.42,
        "color": (42, 112, 208),
        "accent": (72, 162, 98),
        "atmo": (115, 192, 255),
        "has_rings": False,
        "ring_color": (200, 200, 200),
        "spin_speed": 0.9,
    },
    {
        "id": "rocky",
        "name": "Mars",
        "kind": "planet",
        "subtype": "rocky",
        "mass": 2.1,
        "radius": 0.34,
        "color": (196, 94, 58),
        "accent": (136, 56, 36),
        "atmo": (235, 148, 108),
        "has_rings": False,
        "ring_color": (180, 140, 110),
        "spin_speed": 0.75,
    },
    {
        "id": "gas",
        "name": "Saturn",
        "kind": "planet",
        "subtype": "gas",
        "mass": 12.0,
        "radius": 0.78,
        "color": (222, 186, 134),
        "accent": (176, 126, 82),
        "atmo": (244, 214, 168),
        "has_rings": True,
        "ring_color": (218, 194, 152),
        "spin_speed": 1.35,
    },
    {
        "id": "ice",
        "name": "Neptune",
        "kind": "planet",
        "subtype": "ice",
        "mass": 8.5,
        "radius": 0.60,
        "color": (56, 136, 224),
        "accent": (98, 198, 248),
        "atmo": (135, 224, 255),
        "has_rings": True,
        "ring_color": (135, 196, 238),
        "spin_speed": 1.15,
    },
    {
        "id": "lava",
        "name": "Lava",
        "kind": "planet",
        "subtype": "lava",
        "mass": 3.8,
        "radius": 0.40,
        "color": (52, 28, 30),
        "accent": (255, 96, 26),
        "atmo": (255, 136, 52),
        "has_rings": False,
        "ring_color": (255, 120, 40),
        "spin_speed": 0.65,
    },
    {
        "id": "star",
        "name": "Star",
        "kind": "star",
        "subtype": "star",
        "mass": 26.0,
        "radius": 0.92,
        "color": (255, 240, 195),
        "accent": (255, 185, 92),
        "atmo": (255, 212, 132),
        "has_rings": False,
        "ring_color": (255, 210, 120),
        "spin_speed": 0.45,
    },
    {
        "id": "neutron",
        "name": "Pulsar",
        "kind": "neutron",
        "subtype": "neutron",
        "mass": 38.0,
        "radius": 0.30,
        "color": (215, 245, 255),
        "accent": (115, 215, 255),
        "atmo": (140, 228, 255),
        "has_rings": False,
        "ring_color": (130, 220, 255),
        "spin_speed": 5.5,
    },
    {
        "id": "bh",
        "name": "Black Hole",
        "kind": "bh",
        "subtype": "bh",
        "mass": 52.0,
        "radius": 0.64,
        "color": (4, 4, 8),
        "accent": (255, 165, 82),
        "atmo": (255, 238, 205),
        "has_rings": True,
        "ring_color": (255, 165, 78),
        "spin_speed": 2.4,
    },
    {
        "id": "repulsor",
        "name": "Repulsor",
        "kind": "repulsor",
        "subtype": "repulsor",
        "mass": -28.0,
        "radius": 0.46,
        "color": (24, 34, 68),
        "accent": (90, 240, 255),
        "atmo": (180, 120, 255),
        "has_rings": False,
        "ring_color": (120, 220, 255),
        "spin_speed": 1.6,
    },
]

CATALOG_BY_ID = {item["id"]: item for item in SPAWN_CATALOG}


def make_body(type_id, pos, vel=None, mass=None, tilt=None):
    spec = CATALOG_BY_ID[type_id]
    rng = np.random.default_rng()
    if tilt is None:
        tilt = (
            float(rng.uniform(-0.55, 0.55)),
            float(rng.uniform(-0.55, 0.55)),
            float(rng.uniform(0.0, math.pi)),
        )
    r_mat = rot_mat(*tilt)
    jet_axis = normalize(r_mat @ np.array([0.0, 0.0, 1.0], dtype=np.float32))
    return {
        "id": spec["id"],
        "name": spec["name"],
        "kind": spec["kind"],
        "subtype": spec["subtype"],
        "pos": np.array(pos, dtype=np.float32),
        "vel": np.zeros(3, dtype=np.float32) if vel is None else np.array(vel, dtype=np.float32),
        "mass": float(spec["mass"] if mass is None else mass),
        "radius": float(spec["radius"]),
        "color": spec["color"],
        "accent": spec["accent"],
        "atmo": spec["atmo"],
        "has_rings": bool(spec["has_rings"]),
        "ring_color": spec["ring_color"],
        "ring_mat": r_mat,
        "jet_axis": jet_axis,
        "spin": float(rng.uniform(0.0, 2.0 * math.pi)),
        "spin_speed": float(spec["spin_speed"]),
        "trail": [np.array(pos, dtype=np.float32)],
        "trail_acc": 0.0,
        "absorbed": 0,
    }


def make_core(pos, vel, mass):
    return {
        "id": "core",
        "name": "Core",
        "kind": "core",
        "subtype": "core",
        "pos": np.array(pos, dtype=np.float32),
        "vel": np.array(vel, dtype=np.float32),
        "mass": float(mass),
        "radius": 0.75,
        "color": (255, 236, 198),
        "accent": (255, 200, 130),
        "atmo": (255, 224, 172),
        "has_rings": False,
        "ring_color": (255, 220, 160),
        "ring_mat": np.eye(3, dtype=np.float32),
        "jet_axis": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "spin": 0.0,
        "spin_speed": 0.0,
        "trail": [],
        "trail_acc": 0.0,
        "absorbed": 0,
    }


def compute_orbit_velocity(spawn_pos, bodies):
    best = None
    best_pull = -1.0
    for b in bodies:
        if b["mass"] <= 0.0:
            continue
        d = b["pos"] - spawn_pos
        dist2 = float(d @ d) + SOFTENING * SOFTENING
        pull = b["mass"] / dist2
        if pull > best_pull:
            best_pull = pull
            best = b

    if best is None:
        return np.zeros(3, dtype=np.float32)

    r_vec = spawn_pos - best["pos"]
    r = float(np.linalg.norm(r_vec))
    if r < 0.25:
        return best["vel"].copy()

    r2 = r * r + SOFTENING * SOFTENING
    speed = r * math.sqrt((G * best["mass"]) / (r2 * math.sqrt(r2)))

    up_ref = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent = np.cross(up_ref, r_vec)
    if float(np.linalg.norm(tangent)) < 1e-4:
        tangent = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), r_vec)
    tangent = normalize(tangent)

    return (best["vel"] + tangent * speed).astype(np.float32)
