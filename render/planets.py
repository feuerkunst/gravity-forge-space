import sys
from pathlib import Path
import numpy as np
import pygame
from config import W, H, FOCAL
from sim.math3d import normalize, rot_mat
from render.obj_mesh import ObjMesh


class PlanetRenderer:
    def __init__(self):
        self.glow_cache = {}
        self.meshes = self._load_meshes()

    def _load_meshes(self):
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            base_dir = Path(sys._MEIPASS)
        else:
            base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "assets" / "models"
        specs = [
            ("earth", "earth.obj", "earth", "mean", False),
            ("rocky", "rocky.obj", "rocky", "mean", False),
            ("gas", "gas.obj", "gas", "mean", False),
            ("ice", "ice.obj", "ice", "mean", False),
            ("lava", "lava.obj", "lava", "mean", False),
            ("star", "star.obj", "star", "mean", False),
            ("neutron", "neutron.obj", "neutron", "mean", False),
            ("repulsor", "repulsor.obj", "repulsor", "max", False),
            ("bh_disk", "bh_disk.obj", "bh_disk", "max", True),
            ("planet_ring", "planet_ring.obj", "planet_ring", "max", True),
        ]

        missing = [fname for _, fname, *_ in specs if not (models_dir / fname).exists()]
        if missing:
            try:
                from make_obj import generate_all
                generate_all(models_dir)
            except Exception as exc:
                print(f"[PlanetRenderer] Failed to auto-generate OBJ models: {exc}")

        loaded = {}
        for key, fname, style, norm_mode, dbl in specs:
            p = models_dir / fname
            if p.exists():
                loaded[key] = ObjMesh.load(p, style=style, normalize_mode=norm_mode, double_sided=dbl)
            else:
                loaded[key] = None
        return loaded

    def _get_glow_surf(self, color, size, strength=0.42):
        size = max(16, (size // 8) * 8)
        key = (color, size, round(strength, 2))
        if key in self.glow_cache:
            return self.glow_cache[key]

        surf = pygame.Surface((size, size))
        lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        gx, gy = np.meshgrid(lin, lin)
        r = np.sqrt(gx * gx + gy * gy)
        falloff = (np.clip(1.0 - r, 0.0, 1.0) ** 3.2) * strength

        rgb = np.empty((size, size, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(color[0] * falloff, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip(color[1] * falloff, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip(color[2] * falloff, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(surf, rgb)

        if len(self.glow_cache) > 64:
            self.glow_cache.clear()
        self.glow_cache[key] = surf
        return surf

    def _world_light_dir(self, body, bodies):
        best_l = None
        best_score = -1.0
        for other in bodies:
            if other is body or other["kind"] not in ("star", "core", "neutron"):
                continue
            d = other["pos"] - body["pos"]
            dist2 = float(d @ d) + 1.0
            score = other["mass"] / dist2
            if score > best_score:
                best_score = score
                best_l = d
        if best_l is None:
            return normalize(np.array([0.55, 0.52, -0.65], dtype=np.float32))
        return normalize(best_l)

    def draw_orbits(self, screen, bodies, cam_pos, cam_basis):
        draw_lines = pygame.draw.lines
        for b in bodies:
            tr = b["trail"]
            n = len(tr)
            if n < 2:
                continue
            pts = np.asarray(tr, dtype=np.float32)
            cam_p = (pts - cam_pos[None, :]) @ cam_basis
            zs = cam_p[:, 2]
            valid = zs > 0.6
            if np.count_nonzero(valid) < 2:
                continue
            inv_z = FOCAL / np.maximum(zs, 0.6)
            xs = (cam_p[:, 0] * inv_z + W * 0.5).astype(np.int32).tolist()
            ys = (cam_p[:, 1] * inv_z + H * 0.5).astype(np.int32).tolist()
            val_l = valid.tolist()
            base_c = b["atmo"]

            buckets = 4
            step = max(1, n // buckets)
            for k in range(buckets):
                i0 = k * step
                i1 = n if k == buckets - 1 else min(n, (k + 1) * step + 1)
                if i1 - i0 < 2:
                    continue
                f = (k + 1) / buckets * 0.58
                col = (int(base_c[0] * f), int(base_c[1] * f), int(base_c[2] * f))
                seg = []
                for i in range(i0, i1):
                    x, y = xs[i], ys[i]
                    if val_l[i] and -300 <= x <= W + 300 and -300 <= y <= H + 300:
                        seg.append((x, y))
                    else:
                        if len(seg) >= 2:
                            draw_lines(screen, col, False, seg, 1)
                        seg = []
                if len(seg) >= 2:
                    draw_lines(screen, col, False, seg, 1)

    def draw_bodies(self, screen, bodies, cam_pos, fwd, right, up, show_orbits=True):
        cam_basis = np.stack([right, -up, fwd], axis=1).astype(np.float32)

        if show_orbits:
            self.draw_orbits(screen, bodies, cam_pos, cam_basis)

        order = []
        for b in bodies:
            d = b["pos"] - cam_pos
            bz = float(d @ fwd)
            if bz > 0.6:
                order.append((bz, d, b))
        order.sort(key=lambda item: item[0], reverse=True)

        scale_mul = 1.75

        for bz, d, b in order:
            inv_z = FOCAL / bz
            bx = int((d @ right) * inv_z + W * 0.5)
            by = int(-(d @ up) * inv_z + H * 0.5)
            if bx < -280 or bx > W + 280 or by < -280 or by > H + 280:
                continue

            if b["kind"] == "core":
                gsz = max(28, min(112, int(2.2 * inv_z)))
                glow = self._get_glow_surf((255, 205, 140), gsz, strength=0.35)
                screen.blit(glow, (bx - gsz // 2, by - gsz // 2), special_flags=pygame.BLEND_RGB_ADD)
                continue

            draw_r = b["radius"] * scale_mul
            rad = max(3, min(130, int(draw_r * inv_z)))
            light_dir = self._world_light_dir(b, bodies)

            if b["kind"] == "bh":
                gsz = max(40, min(220, int(rad * 3.2)))
                glow = self._get_glow_surf((235, 128, 48), gsz, strength=0.38)
                screen.blit(glow, (bx - gsz // 2, by - gsz // 2), special_flags=pygame.BLEND_RGB_ADD)

                disk_mesh = self.meshes.get("bh_disk")
                disk_rot = b["ring_mat"] @ rot_mat(0.0, b["spin"] * 0.6, 0.0)
                if disk_mesh is not None and rad >= 5:
                    disk_mesh.draw(
                        screen, b["pos"], draw_r * 3.15, disk_rot,
                        cam_pos, cam_basis, light_dir,
                        z_split=("back", bz), doppler=True,
                    )

                pygame.draw.circle(screen, (255, 185, 108), (bx, by), rad + 2, 2)
                pygame.draw.circle(screen, (255, 242, 214), (bx, by), rad + 1, 1)
                pygame.draw.circle(screen, (2, 2, 5), (bx, by), rad)

                if disk_mesh is not None and rad >= 5:
                    disk_mesh.draw(
                        screen, b["pos"], draw_r * 3.15, disk_rot,
                        cam_pos, cam_basis, light_dir,
                        z_split=("front", bz), doppler=True,
                    )
                continue

            if rad < 4:
                if b["kind"] in ("star", "neutron"):
                    gsz = max(16, int(rad * 4.5))
                    glow = self._get_glow_surf(b["atmo"], gsz, strength=0.48)
                    screen.blit(glow, (bx - gsz // 2, by - gsz // 2), special_flags=pygame.BLEND_RGB_ADD)
                pygame.draw.circle(screen, b["color"], (bx, by), rad)
                pygame.draw.circle(screen, b["atmo"], (bx - 1, by - 1), max(1, rad // 2))
                continue

            spin_m = b["ring_mat"] @ rot_mat(0.0, b["spin"], 0.0)
            mesh = self.meshes.get(b["subtype"])

            if b["kind"] == "neutron":
                gsz = max(24, min(140, int(rad * 4.2)))
                glow = self._get_glow_surf((110, 210, 255), gsz, strength=0.42)
                screen.blit(glow, (bx - gsz // 2, by - gsz // 2), special_flags=pygame.BLEND_RGB_ADD)
                beam_len = draw_r * 10.5
                for sign in (1.0, -1.0):
                    pt = b["pos"] + b["jet_axis"] * (beam_len * sign)
                    dp = pt - cam_pos
                    pz = float(dp @ fwd)
                    if pz > 0.6:
                        px = int((dp @ right) * (FOCAL / pz) + W * 0.5)
                        py = int(-(dp @ up) * (FOCAL / pz) + H * 0.5)
                        if abs(px) < W + 400 and abs(py) < H + 400:
                            pygame.draw.line(screen, (145, 228, 255), (bx, by), (px, py), 2)
                if mesh is not None:
                    mesh.draw(screen, b["pos"], draw_r, spin_m, cam_pos, cam_basis, light_dir, atmo_color=b["atmo"])
                else:
                    pygame.draw.circle(screen, b["color"], (bx, by), rad)

            elif b["kind"] == "star":
                gsz = max(40, min(240, int(rad * 3.5)))
                glow = self._get_glow_surf((255, 190, 96), gsz, strength=0.48)
                screen.blit(glow, (bx - gsz // 2, by - gsz // 2), special_flags=pygame.BLEND_RGB_ADD)
                if mesh is not None:
                    mesh.draw(screen, b["pos"], draw_r, spin_m, cam_pos, cam_basis, light_dir, atmo_color=b["atmo"])
                else:
                    pygame.draw.circle(screen, b["color"], (bx, by), rad)

            elif b["kind"] == "repulsor":
                gsz = max(24, min(130, int(rad * 2.8)))
                glow = self._get_glow_surf((95, 195, 255), gsz, strength=0.30)
                screen.blit(glow, (bx - gsz // 2, by - gsz // 2), special_flags=pygame.BLEND_RGB_ADD)
                if mesh is not None:
                    mesh.draw(screen, b["pos"], draw_r, spin_m, cam_pos, cam_basis, light_dir, atmo_color=b["atmo"])
                else:
                    pygame.draw.circle(screen, b["accent"], (bx, by), rad, 2)

            else:
                ring_mesh = self.meshes.get("planet_ring") if b["has_rings"] else None
                ring_tint = (155, 205, 255) if b["subtype"] == "ice" else (248, 236, 212)
                ring_scale = draw_r * (2.35 if b["subtype"] == "gas" else 1.95)

                if ring_mesh is not None:
                    ring_mesh.draw(
                        screen, b["pos"], ring_scale, b["ring_mat"],
                        cam_pos, cam_basis, light_dir,
                        tint_color=ring_tint, z_split=("back", bz),
                    )

                if mesh is not None:
                    mesh.draw(screen, b["pos"], draw_r, spin_m, cam_pos, cam_basis, light_dir, atmo_color=b["atmo"])
                else:
                    pygame.draw.circle(screen, b["color"], (bx, by), rad)

                if ring_mesh is not None:
                    ring_mesh.draw(
                        screen, b["pos"], ring_scale, b["ring_mat"],
                        cam_pos, cam_basis, light_dir,
                        tint_color=ring_tint, z_split=("front", bz),
                    )
