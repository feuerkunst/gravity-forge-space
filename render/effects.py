import numpy as np
import pygame
from config import W, H, FOCAL, G, SOFTENING


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


class ImpactSystem:
    def __init__(self, max_debris=720):
        self.max_debris = max_debris
        self.d_pos = np.zeros((max_debris, 3), dtype=np.float32)
        self.d_vel = np.zeros((max_debris, 3), dtype=np.float32)
        self.d_age = np.zeros(max_debris, dtype=np.float32)
        self.d_life = np.ones(max_debris, dtype=np.float32)
        self.d_col = np.zeros((max_debris, 3), dtype=np.float32)
        self.d_count = 0

        self.flashes = []
        self._flash_cache = {}
        self._rng = np.random.default_rng()

    def clear(self):
        self.d_count = 0
        self.flashes.clear()

    def spawn(self, impact):
        pos = impact["pos"]
        vel = impact["vel"]
        rad = max(0.25, impact["radius"])
        rel_spd = impact.get("rel_speed", 3.0)
        is_bh = impact.get("is_bh", False)

        flash_col = (255, 165, 85) if is_bh else (255, 224, 170)
        ring_col = (255, 138, 62) if is_bh else impact.get("atmo", (255, 185, 115))

        self.flashes.append({
            "pos": pos.copy(),
            "vel": vel.copy(),
            "age": 0.0,
            "dur": min(1.05, 0.58 + rad * 0.32),
            "max_r": rad * (6.2 if is_bh else 4.8),
            "flash_col": flash_col,
            "ring_col": ring_col,
        })
        if len(self.flashes) > 24:
            self.flashes.pop(0)

        n_new = int(min(75, max(42, 36 + rad * 28)))
        if self.d_count + n_new > self.max_debris:
            keep = max(0, self.max_debris - n_new)
            if keep > 0:
                shift = self.d_count - keep
                self.d_pos[:keep] = self.d_pos[shift:self.d_count]
                self.d_vel[:keep] = self.d_vel[shift:self.d_count]
                self.d_age[:keep] = self.d_age[shift:self.d_count]
                self.d_life[:keep] = self.d_life[shift:self.d_count]
                self.d_col[:keep] = self.d_col[shift:self.d_count]
            self.d_count = keep

        i0 = self.d_count
        i1 = i0 + n_new

        dirs = self._rng.normal(0.0, 1.0, size=(n_new, 3)).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-6

        base_eject = max(2.4, min(11.0, 2.2 + rel_spd * 0.55 + rad * 1.8))
        speeds = self._rng.uniform(base_eject * 0.35, base_eject * 1.35, size=(n_new, 1)).astype(np.float32)
        jitter = self._rng.uniform(0.05, rad * 0.55, size=(n_new, 1)).astype(np.float32)

        self.d_pos[i0:i1] = pos[None, :] + dirs * jitter
        self.d_vel[i0:i1] = vel[None, :] + dirs * speeds
        self.d_age[i0:i1] = 0.0
        self.d_life[i0:i1] = self._rng.uniform(0.85, 2.25, size=n_new).astype(np.float32)

        c_src = np.array(impact.get("color", (255, 185, 110)), dtype=np.float32) / 255.0
        hot = np.array([1.0, 0.62, 0.24], dtype=np.float32)
        mix = self._rng.uniform(0.45, 0.85, size=(n_new, 1)).astype(np.float32)
        self.d_col[i0:i1] = c_src[None, :] * (1.0 - mix) + hot[None, :] * mix
        self.d_count = i1

    def update(self, dt, bodies):
        if dt <= 0.0:
            return

        if self.flashes:
            alive_f = []
            for f in self.flashes:
                f["age"] += dt
                if f["age"] < f["dur"]:
                    f["pos"] += f["vel"] * dt
                    alive_f.append(f)
            self.flashes = alive_f

        n = self.d_count
        if n == 0:
            return

        dt_f = np.float32(dt)
        self.d_age[:n] += dt_f
        alive = self.d_age[:n] < self.d_life[:n]
        n_alive = int(np.count_nonzero(alive))
        if n_alive == 0:
            self.d_count = 0
            return
        if n_alive < n:
            self.d_pos[:n_alive] = self.d_pos[:n][alive]
            self.d_vel[:n_alive] = self.d_vel[:n][alive]
            self.d_age[:n_alive] = self.d_age[:n][alive]
            self.d_life[:n_alive] = self.d_life[:n][alive]
            self.d_col[:n_alive] = self.d_col[:n][alive]
            self.d_count = n_alive
            n = n_alive

        p = self.d_pos[:n]
        v = self.d_vel[:n]
        eps2 = np.float32(SOFTENING * SOFTENING * 2.0)

        for b in bodies:
            d = b["pos"][None, :] - p
            dist2 = np.sum(d * d, axis=1) + eps2
            inv_d3 = np.float32(G * b["mass"]) / (dist2 * np.sqrt(dist2))
            v += d * (inv_d3[:, None] * dt_f)

        p += v * dt_f

    def _get_flash_surf(self, color, size, lvl):
        size = max(24, min(260, (size // 12) * 12))
        key = (color, size, lvl)
        surf = self._flash_cache.get(key)
        if surf is not None:
            return surf

        surf = pygame.Surface((size, size))
        lin = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        gx, gy = np.meshgrid(lin, lin)
        r = np.sqrt(gx * gx + gy * gy)
        strength = (lvl / 8.0) * 0.85
        falloff = (np.clip(1.0 - r, 0.0, 1.0) ** 2.6) * strength

        rgb = np.empty((size, size, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.clip(color[0] * falloff, 0, 255).astype(np.uint8)
        rgb[:, :, 1] = np.clip(color[1] * falloff, 0, 255).astype(np.uint8)
        rgb[:, :, 2] = np.clip(color[2] * falloff, 0, 255).astype(np.uint8)
        pygame.surfarray.blit_array(surf, rgb)

        if len(self._flash_cache) > 96:
            self._flash_cache.clear()
        self._flash_cache[key] = surf
        return surf

    def draw(self, screen, cam_pos, fwd, right, up):
        n = self.d_count
        if n == 0 and not self.flashes:
            return

        cam_basis = np.stack([right, -up, fwd], axis=1).astype(np.float32)

        if n > 0:
            head = self.d_pos[:n]
            tail = head - self.d_vel[:n] * np.float32(0.05)

            ch = (head - cam_pos[None, :]) @ cam_basis
            ct = (tail - cam_pos[None, :]) @ cam_basis
            zh = ch[:, 2]
            zt = ct[:, 2]
            vis = (zh > 0.6) & (zt > 0.6)

            if np.any(vis):
                ch_v = ch[vis]
                ct_v = ct[vis]
                inv_h = FOCAL / ch_v[:, 2]
                inv_t = FOCAL / ct_v[:, 2]

                xh = (ch_v[:, 0] * inv_h + W * 0.5).astype(np.int32).tolist()
                yh = (ch_v[:, 1] * inv_h + H * 0.5).astype(np.int32).tolist()
                xt = (ct_v[:, 0] * inv_t + W * 0.5).astype(np.int32).tolist()
                yt = (ct_v[:, 1] * inv_t + H * 0.5).astype(np.int32).tolist()

                t_norm = np.clip(self.d_age[:n][vis] / self.d_life[:n][vis], 0.0, 1.0)
                fade = (1.0 - t_norm) ** 1.35
                white_hot = np.clip(1.0 - t_norm * 3.2, 0.0, 1.0)[:, None]
                ember = np.array([0.85, 0.22, 0.06], dtype=np.float32)[None, :]
                cool_mix = np.clip((t_norm - 0.35) * 1.4, 0.0, 0.75)[:, None]

                base_c = self.d_col[:n][vis] * (1.0 - cool_mix) + ember * cool_mix
                col_f = (base_c * (1.0 - white_hot) + white_hot) * fade[:, None]
                cols = (np.clip(col_f, 0.0, 1.0) * 255.0).astype(np.uint8).tolist()
                z_list = ch_v[:, 2].tolist()

                draw_line = pygame.draw.line
                for x0, y0, x1, y1, col, z in zip(xh, yh, xt, yt, cols, z_list):
                    if -60 <= x0 <= W + 60 and -60 <= y0 <= H + 60:
                        w_px = 2 if z < 26.0 else 1
                        draw_line(screen, col, (x0, y0), (x1, y1), w_px)

        for f in self.flashes:
            d = f["pos"] - cam_pos
            z = float(d @ fwd)
            if z <= 0.6:
                continue
            inv_z = FOCAL / z
            sx = int((d @ right) * inv_z + W * 0.5)
            sy = int(-(d @ up) * inv_z + H * 0.5)
            if sx < -260 or sx > W + 260 or sy < -260 or sy > H + 260:
                continue

            p = min(1.0, f["age"] / f["dur"])
            ease = 1.0 - (1.0 - p) ** 2.4
            fade_core = (1.0 - p) ** 2.0
            fade_ring = (1.0 - p) ** 1.45

            lvl = int(round(fade_core * 8.0))
            if lvl >= 1:
                gsz = max(28, min(260, int(f["max_r"] * (0.45 + 0.55 * ease) * inv_z * 2.2)))
                glow = self._get_flash_surf(f["flash_col"], gsz, lvl)
                screen.blit(glow, (sx - glow.get_width() // 2, sy - glow.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)

            ring_r = int(f["max_r"] * (0.18 + 0.82 * ease) * inv_z)
            if 3 <= ring_r <= 420:
                rc = f["ring_col"]
                col_r = (
                    int(rc[0] * fade_ring),
                    int(rc[1] * fade_ring),
                    int(rc[2] * fade_ring),
                )
                pygame.draw.circle(screen, col_r, (sx, sy), ring_r, 2 if p < 0.45 else 1)

            if p < 0.48:
                inner_r = max(2, int(ring_r * 0.58))
                ic = int(255 * ((1.0 - p / 0.48) ** 1.5))
                pygame.draw.circle(screen, (ic, int(ic * 0.88), int(ic * 0.68)), (sx, sy), inner_r, 1)
