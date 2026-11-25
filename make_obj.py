import math
from pathlib import Path


def _norm(x, y, z):
    n = math.sqrt(x * x + y * y + z * z) + 1e-9
    return x / n, y / n, z / n


def _icosphere(subdivisions=3):
    t = (1.0 + math.sqrt(5.0)) * 0.5
    raw = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    verts = [_norm(*p) for p in raw]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]

    for _ in range(subdivisions):
        mid_cache = {}
        new_faces = []

        def midpoint(i1, i2):
            key = (i1, i2) if i1 < i2 else (i2, i1)
            if key in mid_cache:
                return mid_cache[key]
            x1, y1, z1 = verts[i1]
            x2, y2, z2 = verts[i2]
            verts.append(_norm((x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5))
            idx = len(verts) - 1
            mid_cache[key] = idx
            return idx

        for a, b, c in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces.append((a, ab, ca))
            new_faces.append((b, bc, ab))
            new_faces.append((c, ca, bc))
            new_faces.append((ab, bc, ca))
        faces = new_faces

    fixed = []
    for a, b, c in faces:
        ax, ay, az = verts[a]
        bx, by, bz = verts[b]
        cx, cy, cz = verts[c]
        e1x, e1y, e1z = bx - ax, by - ay, bz - az
        e2x, e2y, e2z = cx - ax, cy - ay, cz - az
        nx = e1y * e2z - e1z * e2y
        ny = e1z * e2x - e1x * e2z
        nz = e1x * e2y - e1y * e2x
        if nx * (ax + bx + cx) + ny * (ay + by + cy) + nz * (az + bz + cz) < 0.0:
            fixed.append((a + 1, c + 1, b + 1))
        else:
            fixed.append((a + 1, b + 1, c + 1))
    return verts, fixed


def _fbm3(x, y, z, seed=0.0):
    val = 0.0
    amp = 0.55
    freq = 2.1
    for k in range(4):
        s = seed + k * 1.73
        val += amp * math.sin(x * freq + s) * math.cos(y * freq * 1.13 - s * 0.7) * math.sin(z * freq * 0.91 + s * 1.3)
        val += amp * 0.45 * math.cos((x - z) * freq * 0.85 + y * freq * 0.6 + s * 2.1)
        freq *= 1.92
        amp *= 0.52
    return val


def _write_obj(path: Path, name: str, verts, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# Gravity Forge 3D Mesh: {name}\n", f"o {name}\n"]
    for vx, vy, vz in verts:
        lines.append(f"v {vx:.5f} {vy:.5f} {vz:.5f}\n")
    for f1, f2, f3 in faces:
        lines.append(f"f {f1} {f2} {f3}\n")
    path.write_text("".join(lines), encoding="utf-8")


def _make_annulus_disk(bands, sectors=64, warp=0.0):
    verts = []
    faces = []
    for r_in, r_out, h_in, h_out in bands:
        base = len(verts)
        for k in range(sectors):
            ang = (k / sectors) * 2.0 * math.pi
            ca, sa = math.cos(ang), math.sin(ang)
            w = warp * math.sin(ang * 2.0) * (r_out - r_in)
            verts.append((ca * r_in, h_in + w * 0.25, sa * r_in))
            verts.append((ca * r_out, h_out + w, sa * r_out))
            verts.append((ca * r_in, -h_in + w * 0.25, sa * r_in))
            verts.append((ca * r_out, -h_out + w, sa * r_out))
        for k in range(sectors):
            kn = (k + 1) % sectors
            t_in0, t_out0, b_in0, b_out0 = base + k * 4 + 1, base + k * 4 + 2, base + k * 4 + 3, base + k * 4 + 4
            t_in1, t_out1, b_in1, b_out1 = base + kn * 4 + 1, base + kn * 4 + 2, base + kn * 4 + 3, base + kn * 4 + 4
            faces.append((t_in0, t_out1, t_out0))
            faces.append((t_in0, t_in1, t_out1))
            faces.append((b_in0, b_out0, b_out1))
            faces.append((b_in0, b_out1, b_in1))
    return verts, faces


def _make_repulsor():
    base_v, base_f = _icosphere(subdivisions=2)
    verts = []
    for x, y, z in base_v:
        eq = math.exp(-((y * 5.2) ** 2))
        facet = abs(math.sin(x * 4.5) * math.cos(z * 4.5))
        r = 0.90 + 0.16 * facet + 0.22 * eq
        verts.append((x * r, y * r * 1.12, z * r))
    ring_v, ring_f = _make_annulus_disk([(1.28, 1.48, 0.045, 0.03)], sectors=36)
    offset = len(verts)
    verts.extend(ring_v)
    faces = list(base_f) + [(a + offset, b + offset, c + offset) for a, b, c in ring_f]
    return verts, faces


def generate_all(models_dir: Path):
    base_v, base_f = _icosphere(subdivisions=3)

    # 1. Earth: flat oceans + sculpted continents & mountain ridges
    earth_v = []
    for x, y, z in base_v:
        h = _fbm3(x, y, z, seed=1.3)
        if h > 0.03:
            elev = 0.012 + (h - 0.03) * 0.11 + max(0.0, h - 0.34) * 0.14
        else:
            elev = 0.0
        r = 1.0 + elev
        earth_v.append((x * r, y * r, z * r))
    _write_obj(models_dir / "earth.obj", "Earth", earth_v, base_f)

    # 2. Mars: dusty relief + sculpted impact craters with raised rims
    craters = [
        (_norm(0.6, 0.25, 0.75), 0.36, 0.065),
        (_norm(-0.7, -0.15, 0.65), 0.42, 0.070),
        (_norm(0.15, -0.65, -0.72), 0.32, 0.055),
        (_norm(-0.4, 0.62, -0.66), 0.38, 0.062),
        (_norm(0.85, -0.35, -0.35), 0.28, 0.050),
        (_norm(-0.82, 0.35, 0.25), 0.30, 0.052),
    ]
    mars_v = []
    for x, y, z in base_v:
        h = 0.035 * _fbm3(x, y, z, seed=4.7)
        for (cx, cy, cz), rad, depth in craters:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            if d < rad:
                t = d / rad
                h -= depth * (1.0 - t * t)
            elif d < rad * 1.28:
                t = (d - rad) / (rad * 0.28)
                h += depth * 0.34 * math.sin(t * math.pi)
        r = 1.0 + h
        mars_v.append((x * r, y * r, z * r))
    _write_obj(models_dir / "rocky.obj", "Mars", mars_v, base_f)

    # 3. Gas Giant (Saturn): smooth oblate spheroid
    gas_v = []
    for x, y, z in base_v:
        band = 0.008 * math.sin(y * 14.0)
        r = 1.0 + band
        gas_v.append((x * r, y * r * 0.89, z * r))
    _write_obj(models_dir / "gas.obj", "Saturn", gas_v, base_f)

    # 4. Ice Giant (Neptune): subtle oblate spheroid
    ice_v = []
    for x, y, z in base_v:
        band = 0.007 * math.cos(y * 10.0)
        r = 1.0 + band
        ice_v.append((x * r, y * r * 0.94, z * r))
    _write_obj(models_dir / "ice.obj", "Neptune", ice_v, base_f)

    # 5. Lava world: volcanic crust plates + deep magma rifts
    lava_v = []
    for x, y, z in base_v:
        n1 = _fbm3(x * 1.3, y * 1.3, z * 1.3, seed=9.1)
        rift = abs(n1)
        if rift < 0.14:
            elev = -0.055 * (1.0 - rift / 0.14)
        else:
            elev = 0.015 + 0.045 * max(0.0, n1)
        r = 1.0 + elev
        lava_v.append((x * r, y * r, z * r))
    _write_obj(models_dir / "lava.obj", "LavaPlanet", lava_v, base_f)

    # 6. Star: subtle convective granulation
    star_v = []
    for x, y, z in base_v:
        g = _fbm3(x * 1.8, y * 1.8, z * 1.8, seed=13.5)
        r = 1.0 + 0.032 * g
        star_v.append((x * r, y * r, z * r))
    _write_obj(models_dir / "star.obj", "Star", star_v, base_f)

    # 7. Pulsar / Neutron star: compact high-energy core
    ns_v, ns_f = _icosphere(subdivisions=2)
    _write_obj(models_dir / "neutron.obj", "Pulsar", ns_v, ns_f)

    # 8. Repulsor: faceted sci-fi core + stabilizer ring
    rep_v, rep_f = _make_repulsor()
    _write_obj(models_dir / "repulsor.obj", "Repulsor", rep_v, rep_f)

    # 9. Black Hole 3D Accretion Disk (multi-band with inner hot edge)
    bh_v, bh_f = _make_annulus_disk(
        [
            (0.42, 0.56, 0.024, 0.018),
            (0.56, 0.78, 0.018, 0.012),
            (0.78, 1.00, 0.012, 0.005),
        ],
        sectors=64,
        warp=0.05,
    )
    _write_obj(models_dir / "bh_disk.obj", "AccretionDisk", bh_v, bh_f)

    # 10. Planetary 3D Rings (with Cassini division gap)
    pr_v, pr_f = _make_annulus_disk(
        [
            (0.58, 0.76, 0.008, 0.006),
            (0.80, 1.00, 0.006, 0.004),
        ],
        sectors=56,
        warp=0.0,
    )
    _write_obj(models_dir / "planet_ring.obj", "PlanetaryRing", pr_v, pr_f)

    print(f"Generated 10 OBJ models in {models_dir}")


if __name__ == "__main__":
    generate_all(Path(__file__).resolve().parent / "assets" / "models")
