import numpy as np
import pygame
from config import W, H, FOCAL, GW, GH, G_CELLS
from render.effects import box_blur, paint_background_stars


class HDRenderer:
    def __init__(self):
        self.frame = np.empty((W, H, 3), dtype=np.uint8)
        self.trail_buf = np.zeros((GW, GH, 3), dtype=np.float32)
        self.blur_a = np.empty((GW, GH, 3), dtype=np.float32)
        self.blur_b = np.empty((GW, GH, 3), dtype=np.float32)
        self.glow_u8 = np.empty((GW, GH, 3), dtype=np.uint8)
        self.glow_surf = pygame.Surface((GW, GH))
        self.frame_surf = pygame.Surface((W, H))

    def clear_trails(self):
        self.trail_buf.fill(0.0)

    def render(self, screen, pos, vel, base_cols, cam_pos, fwd, right, up, bg_stars, trails_on):
        self.frame[:, :, 0] = 3
        self.frame[:, :, 1] = 5
        self.frame[:, :, 2] = 10

        cam_basis = np.stack([right, -up, fwd], axis=1).astype(np.float32)
        paint_background_stars(self.frame, bg_stars[0], bg_stars[1], cam_basis)

        offset = -(cam_pos @ cam_basis)
        cam_p = pos @ cam_basis
        cam_p += offset

        cz = cam_p[:, 2]
        vis = cz > 0.6

        if trails_on:
            self.trail_buf *= 0.68
        else:
            self.trail_buf.fill(0.0)

        if np.any(vis):
            cp = cam_p[vis]
            czm = cp[:, 2]
            inv_z = FOCAL / czm
            sx = (cp[:, 0] * inv_z + W * 0.5).astype(np.int32)
            sy = (cp[:, 1] * inv_z + H * 0.5).astype(np.int32)
            ok = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)

            if np.any(ok):
                sx = sx[ok]
                sy = sy[ok]
                cz_ok = czm[ok]
                c_ok = base_cols[vis][ok]

                vm = vel[vis][ok]
                spd2 = vm[:, 0] * vm[:, 0] + vm[:, 1] * vm[:, 1] + vm[:, 2] * vm[:, 2]
                boost = np.clip((spd2 - 14.0) * 0.016, 0.0, 0.55)[:, None]
                fade = np.clip(38.0 / (cz_ok + 5.5), 0.34, 1.38)[:, None]
                c_lit = np.clip(c_ok * (fade * (1.0 + boost)), 0.0, 1.0)

                rgb_u8 = (c_lit * 255.0).astype(np.uint8)
                self.frame[sx, sy] = np.maximum(self.frame[sx, sy], rgb_u8)

                near = cz_ok < 38.0
                if np.any(near):
                    nx = np.minimum(sx[near] + 1, W - 1)
                    ny = np.minimum(sy[near] + 1, H - 1)
                    nc = (rgb_u8[near].astype(np.uint16) * 205 >> 8).astype(np.uint8)
                    self.frame[nx, sy[near]] = np.maximum(self.frame[nx, sy[near]], nc)
                    self.frame[sx[near], ny] = np.maximum(self.frame[sx[near], ny], nc)

                gx = sx >> 2
                gy = sy >> 2
                gflat = gx * GH + gy
                w_g = fade[:, 0] * 0.24
                r_g = np.bincount(gflat, weights=c_lit[:, 0] * w_g, minlength=G_CELLS)
                g_g = np.bincount(gflat, weights=c_lit[:, 1] * w_g, minlength=G_CELLS)
                b_g = np.bincount(gflat, weights=c_lit[:, 2] * w_g, minlength=G_CELLS)
                self.trail_buf[:, :, 0] += r_g.reshape(GW, GH)
                self.trail_buf[:, :, 1] += g_g.reshape(GW, GH)
                self.trail_buf[:, :, 2] += b_g.reshape(GW, GH)

        pygame.surfarray.blit_array(self.frame_surf, self.frame)
        screen.blit(self.frame_surf, (0, 0))

        blurred = box_blur(self.trail_buf, passes=2, buf_a=self.blur_a, buf_b=self.blur_b)
        lum = np.maximum(blurred[:, :, 0], np.maximum(blurred[:, :, 1], blurred[:, :, 2]))
        exp_lum = lum * 1.65
        scale = (exp_lum / (1.0 + exp_lum)) / (lum + 1e-6)
        np.multiply(blurred, scale[:, :, None] * 215.0, out=self.blur_a)
        np.clip(self.blur_a, 0.0, 255.0, out=self.blur_a)
        np.copyto(self.glow_u8, self.blur_a, casting="unsafe")

        pygame.surfarray.blit_array(self.glow_surf, self.glow_u8)
        screen.blit(pygame.transform.smoothscale(self.glow_surf, (W, H)), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
