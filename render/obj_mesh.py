import math
from pathlib import Path
import numpy as np
import pygame
from config import W, H, FOCAL


class ObjMesh:
    def __init__(self, verts: np.ndarray, tris: np.ndarray, style="planet", double_sided=False):
        self.verts = verts.astype(np.float32)
        self.tris = tris.astype(np.int32)
        self.style = style
        self.double_sided = double_sided

        p0 = self.verts[self.tris[:, 0]]
        p1 = self.verts[self.tris[:, 1]]
        p2 = self.verts[self.tris[:, 2]]

        self.face_centers = ((p0 + p1 + p2) / 3.0).astype(np.float32)
        n = np.cross(p1 - p0, p2 - p0)
        n_len = np.linalg.norm(n, axis=1, keepdims=True) + 1e-6
        self.face_normals = (n / n_len).astype(np.float32)
        self.face_d = np.sum(self.face_normals * self.face_centers, axis=1).astype(np.float32)

        self.face_albedo, self.face_emissive, self.face_spec = self._bake_materials(style)

    @classmethod
    def load(cls, path: Path, style="planet", normalize_mode="mean", double_sided=False):
        raw_v = []
        raw_f = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                tag = parts[0]
                if tag == "v" and len(parts) >= 4:
                    raw_v.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif tag == "f" and len(parts) >= 4:
                    idxs = []
                    for tok in parts[1:]:
                        idx_str = tok.split("/")[0]
                        if not idx_str:
                            continue
                        idx = int(idx_str)
                        idxs.append(len(raw_v) + idx if idx < 0 else idx - 1)
                    for k in range(1, len(idxs) - 1):
                        raw_f.append((idxs[0], idxs[k], idxs[k + 1]))

        if not raw_v or not raw_f:
            raise ValueError(f"Empty or invalid OBJ: {path}")

        v = np.asarray(raw_v, dtype=np.float32)
        v = v - v.mean(axis=0, keepdims=True)
        norms = np.linalg.norm(v, axis=1)
        if normalize_mode == "max":
            scale = float(np.max(norms))
            if scale > 1e-6:
                v = v / scale
        elif normalize_mode == "mean":
            scale = float(np.mean(norms))
            if scale > 1e-6:
                v = v / scale

        t = np.asarray(raw_f, dtype=np.int32)
        valid = np.all((t >= 0) & (t < len(v)), axis=1)
        t = t[valid]
        return cls(v, t, style=style, double_sided=double_sided)

    def _bake_materials(self, style):
        nf = len(self.tris)
        c = self.face_centers
        r = np.linalg.norm(c, axis=1)
        r_min, r_max = float(r.min()), float(r.max())
        h = (r - r_min) / max(1e-5, r_max - r_min)
        lat = np.clip(c[:, 1] / np.maximum(r, 1e-5), -1.0, 1.0)
        abs_lat = np.abs(lat)
        lon = np.arctan2(c[:, 2], c[:, 0])

        albedo = np.full((nf, 3), 0.7, dtype=np.float32)
        emissive = np.zeros(nf, dtype=np.float32)
        spec = np.zeros(nf, dtype=np.float32)

        if style == "earth":
            ocean = h < 0.14
            coast = (h >= 0.14) & (h < 0.24)
            plains = (h >= 0.24) & (h < 0.62)
            mountains = h >= 0.62
            desert = plains & (abs_lat < 0.34) & (np.sin(lon * 2.0 + lat * 3.0) > 0.25)
            snow = (abs_lat > 0.80 - 0.08 * h) | (h > 0.84)

            albedo[ocean] = np.array([0.08, 0.27, 0.62], dtype=np.float32)
            albedo[coast] = np.array([0.14, 0.44, 0.56], dtype=np.float32)
            albedo[plains] = np.array([0.20, 0.50, 0.25], dtype=np.float32)
            albedo[desert] = np.array([0.74, 0.62, 0.40], dtype=np.float32)
            albedo[mountains] = np.array([0.46, 0.40, 0.35], dtype=np.float32)
            albedo[snow] = np.array([0.92, 0.95, 0.98], dtype=np.float32)

            spec[ocean] = 0.70
            spec[snow] = 0.25

        elif style == "rocky":
            c1 = np.array([0.44, 0.18, 0.12], dtype=np.float32)
            c2 = np.array([0.78, 0.38, 0.22], dtype=np.float32)
            c3 = np.array([0.90, 0.56, 0.36], dtype=np.float32)
            albedo = c1 * (1.0 - h[:, None]) + c2 * h[:, None]
            rims = h > 0.68
            albedo[rims] = albedo[rims] * 0.5 + c3 * 0.5
            caps = abs_lat > 0.86
            albedo[caps] = np.array([0.90, 0.86, 0.82], dtype=np.float32)

        elif style == "gas":
            wave = 0.5 + 0.5 * np.sin(lat * 16.0 + 0.35 * np.sin(lon * 3.0))
            sub = 0.5 + 0.5 * np.cos(lat * 28.0 - 0.2 * np.cos(lon * 5.0))
            g1 = np.array([0.64, 0.44, 0.28], dtype=np.float32)
            g2 = np.array([0.90, 0.78, 0.58], dtype=np.float32)
            g3 = np.array([0.76, 0.58, 0.38], dtype=np.float32)
            albedo = g1 * (1.0 - wave[:, None]) + g2 * wave[:, None]
            albedo = albedo * 0.75 + g3 * (0.25 * sub[:, None])
            storm = (np.abs(lat + 0.28) < 0.12) & (np.abs(lon - 0.9) < 0.28)
            albedo[storm] = np.array([0.84, 0.38, 0.22], dtype=np.float32)

        elif style == "ice":
            wave = 0.5 + 0.5 * np.sin(lat * 11.0 + 0.3 * np.sin(lon * 2.0))
            i1 = np.array([0.14, 0.34, 0.76], dtype=np.float32)
            i2 = np.array([0.36, 0.74, 0.96], dtype=np.float32)
            albedo = i1 * (1.0 - wave[:, None]) + i2 * wave[:, None]
            wisps = (np.abs(lat - 0.22) < 0.06) & (np.sin(lon * 3.0) > 0.45)
            albedo[wisps] = np.array([0.76, 0.92, 1.0], dtype=np.float32)
            spec[:] = 0.28

        elif style == "lava":
            magma = h < 0.38
            warm = (h >= 0.38) & (h < 0.52)
            albedo[:] = np.array([0.15, 0.11, 0.13], dtype=np.float32) * (0.75 + 0.55 * h[:, None])
            albedo[warm] = np.array([0.52, 0.16, 0.08], dtype=np.float32)
            hot = np.clip((0.38 - h[magma]) / 0.38, 0.0, 1.0)[:, None]
            albedo[magma] = (
                np.array([0.96, 0.30, 0.04], dtype=np.float32) * (1.0 - hot)
                + np.array([1.0, 0.86, 0.30], dtype=np.float32) * hot
            )
            emissive[magma] = 1.0
            emissive[warm] = 0.35

        elif style == "star":
            s1 = np.array([1.0, 0.62, 0.22], dtype=np.float32)
            s2 = np.array([1.0, 0.95, 0.76], dtype=np.float32)
            albedo = s1 * (1.0 - h[:, None]) + s2 * h[:, None]
            emissive[:] = 1.0

        elif style == "neutron":
            albedo[:] = np.array([0.74, 0.94, 1.0], dtype=np.float32)
            emissive[:] = 1.0

        elif style == "repulsor":
            is_ring = np.hypot(c[:, 0], c[:, 2]) > 1.08
            glow_band = (np.abs(c[:, 1]) < 0.16) | is_ring
            albedo[:] = np.array([0.16, 0.22, 0.34], dtype=np.float32)
            albedo[glow_band] = np.array([0.36, 0.88, 1.0], dtype=np.float32)
            albedo[is_ring] = np.array([0.68, 0.52, 1.0], dtype=np.float32)
            emissive[glow_band] = 0.88
            spec[~glow_band] = 0.65

        elif style == "bh_disk":
            r_xz = np.hypot(c[:, 0], c[:, 2])
            t = np.clip((r_xz - 0.42) / 0.58, 0.0, 1.0)[:, None]
            inner = np.array([1.0, 0.94, 0.80], dtype=np.float32)
            mid = np.array([1.0, 0.54, 0.16], dtype=np.float32)
            outer = np.array([0.52, 0.14, 0.06], dtype=np.float32)
            albedo = np.where(
                t < 0.45,
                inner * (1.0 - t / 0.45) + mid * (t / 0.45),
                mid * (1.0 - (t - 0.45) / 0.55) + outer * ((t - 0.45) / 0.55),
            )
            emissive[:] = 1.0

        elif style == "planet_ring":
            r_xz = np.hypot(c[:, 0], c[:, 2])
            band = 0.5 + 0.5 * np.sin(r_xz * 42.0)
            c1 = np.array([0.66, 0.58, 0.46], dtype=np.float32)
            c2 = np.array([0.90, 0.82, 0.68], dtype=np.float32)
            albedo = c1 * (1.0 - band[:, None]) + c2 * band[:, None]

        return albedo, emissive, spec

    def draw(
        self,
        screen,
        world_pos,
        radius,
        rot_m,
        cam_pos,
        cam_basis,
        light_dir,
        atmo_color=(140, 190, 255),
        tint_color=None,
        z_split=None,
        doppler=False,
    ):
        inv_r = 1.0 / max(1e-5, float(radius))
        local_cam = ((cam_pos - world_pos) * inv_r) @ rot_m

        ndotc = self.face_normals @ local_cam - self.face_d
        if self.double_sided:
            front_mask = np.abs(ndotc) > 1e-4
        else:
            front_mask = ndotc > 0.0

        if not np.any(front_mask):
            return

        m_proj = (float(radius) * rot_m.T) @ cam_basis
        offset = (world_pos - cam_pos) @ cam_basis
        cam_v = self.verts @ m_proj
        cam_v += offset

        zs = cam_v[:, 2]
        f_tris = self.tris[front_mask]
        z0 = zs[f_tris[:, 0]]
        z1 = zs[f_tris[:, 1]]
        z2 = zs[f_tris[:, 2]]
        valid_z = (z0 > 0.45) & (z1 > 0.45) & (z2 > 0.45)

        if z_split is not None:
            side, split_z = z_split
            z_avg = (z0 + z1 + z2) * (1.0 / 3.0)
            if side == "back":
                valid_z &= z_avg >= split_z
            else:
                valid_z &= z_avg < split_z

        if not np.any(valid_z):
            return

        sel_idx = np.flatnonzero(front_mask)[valid_z]
        f_tris = self.tris[sel_idx]
        f_z = zs[f_tris[:, 0]] + zs[f_tris[:, 1]] + zs[f_tris[:, 2]]

        order = np.argsort(-f_z)
        sel_idx = sel_idx[order]
        f_tris = f_tris[order]

        safe_z = np.maximum(zs, 0.45)
        inv_z = FOCAL / safe_z
        xs = cam_v[:, 0] * inv_z + (W * 0.5)
        ys = cam_v[:, 1] * inv_z + (H * 0.5)

        f_albedo = self.face_albedo[sel_idx]
        if tint_color is not None:
            t_arr = np.array(tint_color, dtype=np.float32) / 255.0
            f_albedo = f_albedo * t_arr[None, :]

        f_emis = self.face_emissive[sel_idx]
        f_n = self.face_normals[sel_idx]
        if self.double_sided:
            flip = np.where(ndotc[sel_idx] < 0.0, -1.0, 1.0)[:, None]
            f_n = f_n * flip

        lc_len = float(np.linalg.norm(local_cam)) + 1e-6
        local_view = local_cam / lc_len

        if doppler:
            fc = self.face_centers[sel_idx]
            tangent = np.stack([-fc[:, 2], np.zeros_like(fc[:, 0]), fc[:, 0]], axis=-1)
            tn = np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-6
            tangent /= tn
            d_shift = tangent @ local_view
            mul = np.clip(0.92 + 0.42 * d_shift, 0.48, 1.42)[:, None]
            lit = f_albedo * mul
        else:
            local_light = light_dir @ rot_m
            ndotl = f_n @ local_light
            diff = np.clip(ndotl, 0.0, 1.0)
            hemi = 0.06 + 0.05 * np.clip(f_n[:, 1] * 0.5 + 0.5, 0.0, 1.0)
            shade = (hemi + 0.90 * (diff ** 0.90))[:, None]

            v_dot = np.clip(f_n @ local_view, 0.0, 1.0)
            rim = ((1.0 - v_dot) ** 2.5) * np.clip(ndotl + 0.35, 0.0, 1.0)
            atmo = np.array(atmo_color, dtype=np.float32) / 255.0

            lit = f_albedo * shade + atmo[None, :] * (rim[:, None] * 0.24)

            f_spec = self.face_spec[sel_idx]
            if np.any(f_spec > 0.0):
                hv = local_light + local_view
                hv /= float(np.linalg.norm(hv)) + 1e-6
                sp = (np.clip(f_n @ hv, 0.0, 1.0) ** 18.0) * f_spec
                lit = lit + sp[:, None] * 0.42

            if np.any(f_emis > 0.0):
                e = f_emis[:, None]
                if self.style in ("star", "neutron"):
                    limb = (0.68 + 0.32 * (v_dot ** 0.5))[:, None]
                    lit = lit * (1.0 - e) + f_albedo * limb * e
                else:
                    lit = lit * (1.0 - e) + f_albedo * e

        cols_u8 = (np.clip(lit, 0.0, 1.0) * 255.0).astype(np.uint8)

        x_i = xs.astype(np.int32).tolist()
        y_i = ys.astype(np.int32).tolist()
        tris_l = f_tris.tolist()
        cols_l = cols_u8.tolist()
        draw_poly = pygame.draw.polygon
        w_lim = W + 240
        h_lim = H + 240

        for (a, b, c_idx), col in zip(tris_l, cols_l):
            xa, ya = x_i[a], y_i[a]
            xb, yb = x_i[b], y_i[b]
            xc, yc = x_i[c_idx], y_i[c_idx]
            if (xa < -240 and xb < -240 and xc < -240) or (xa > w_lim and xb > w_lim and xc > w_lim):
                continue
            if (ya < -240 and yb < -240 and yc < -240) or (ya > h_lim and yb > h_lim and yc > h_lim):
                continue
            draw_poly(screen, col, ((xa, ya), (xb, yb), (xc, yc)))
