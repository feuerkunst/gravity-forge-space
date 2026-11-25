import numpy as np
import pygame
from config import W, H, FOCAL, CW, CH, ACOLS, AROWS, ACELLS, RAMP


class AsciiRenderer:
    def __init__(self):
        self.atlas = self._make_atlas()
        self.frame_surf = pygame.Surface((W, H))

    def _make_atlas(self):
        font = pygame.font.SysFont("consolas,couriernew,monospace", 12, bold=True)
        atlas = np.zeros((len(RAMP), CH, CW), dtype=np.uint16)
        surf = pygame.Surface((CW, CH))
        for i, ch in enumerate(RAMP):
            if i == 0:
                continue
            surf.fill((0, 0, 0))
            g = font.render(ch, True, (255, 255, 255))
            gw, gh = g.get_size()
            surf.blit(g, ((CW - gw) // 2, (CH - gh) // 2))
            arr = pygame.surfarray.array3d(surf)[:, :, 0].T.astype(np.uint16)
            atlas[i] = np.minimum(256, (arr * 257) >> 8)
        return atlas

    def render(self, screen, pos, base_cols, cam_pos, fwd, right, up):
        rel = pos - cam_pos
        cz = rel @ fwd
        vis = cz > 0.6

        col_grid = np.zeros((AROWS, ACOLS, 3), dtype=np.float32)
        if np.any(vis):
            rm = rel[vis]
            czm = cz[vis]
            inv_z = FOCAL / czm
            cx = np.floor((rm @ right) * inv_z / CW + ACOLS * 0.5).astype(np.int32)
            cy = np.floor(-(rm @ up) * inv_z / CH + AROWS * 0.5).astype(np.int32)
            ok = (cx >= 0) & (cx < ACOLS) & (cy >= 0) & (cy < AROWS)
            if np.any(ok):
                flat = cy[ok] * ACOLS + cx[ok]
                fade = np.clip(28.0 / (czm[ok] + 8.0), 0.18, 1.35)
                c_ok = base_cols[vis][ok]
                r_a = np.bincount(flat, weights=c_ok[:, 0] * fade, minlength=ACELLS)
                g_a = np.bincount(flat, weights=c_ok[:, 1] * fade, minlength=ACELLS)
                b_a = np.bincount(flat, weights=c_ok[:, 2] * fade, minlength=ACELLS)
                raw = np.stack([r_a, g_a, b_a], axis=-1).reshape(AROWS, ACOLS, 3)
                col_grid = 1.0 - np.exp(-raw * 0.55)

        lum = 0.299 * col_grid[:, :, 0] + 0.587 * col_grid[:, :, 1] + 0.114 * col_grid[:, :, 2]
        idx = np.clip((lum * (len(RAMP) - 1) * 1.15).astype(np.int32), 0, len(RAMP) - 1)
        idx[lum < 0.04] = 0
        c_u16 = (np.clip(col_grid, 0.0, 1.0) * 255.0).astype(np.uint16)
        img = ((self.atlas[idx][..., None] * c_u16[:, :, None, None, :]) >> 8).transpose(0, 2, 1, 3, 4).reshape(H, W, 3).astype(np.uint8)
        img[:, :, 0] = np.maximum(img[:, :, 0], 4)
        img[:, :, 1] = np.maximum(img[:, :, 1], 6)
        img[:, :, 2] = np.maximum(img[:, :, 2], 12)
        pygame.surfarray.blit_array(self.frame_surf, img.swapaxes(0, 1))
        screen.blit(self.frame_surf, (0, 0))
