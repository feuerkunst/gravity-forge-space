import time
import numpy as np
import pygame
from config import W, H, FOCAL
from sim.bodies import SPAWN_CATALOG
from render.camera import world_to_screen
from ui.lang import get_text, body_label, preset_label


class HUD:
    def __init__(self, lang="ru"):
        self.font = pygame.font.SysFont("segoeui,arial,tahoma", 14)
        self.bold = pygame.font.SysFont("segoeui,arial,tahoma", 14, bold=True)
        self.title_font = pygame.font.SysFont("segoeui,arial,tahoma", 16, bold=True)
        self.small = pygame.font.SysFont("segoeui,arial,tahoma", 12)
        self.meta = pygame.font.SysFont("consolas,segoeui,arial", 13)

        self.lang = lang
        self._surface_cache = {}
        self.toast = ""
        self.toast_end = 0.0

        self.menu_btn = pygame.Rect(18, 14, 78, 32)
        self.status_rect = pygame.Rect(104, 14, 410, 32)

        self.time_panel = pygame.Rect(W - 266, 14, 248, 32)
        self.btn_slow = pygame.Rect(self.time_panel.x + 8, 18, 26, 24)
        self.btn_fast = pygame.Rect(self.time_panel.x + 88, 18, 26, 24)
        self.btn_pause = pygame.Rect(self.time_panel.x + 120, 18, 52, 24)

        self.card_rect = pygame.Rect(W - 266, 56, 248, 174)
        self.btn_unlock = pygame.Rect(self.card_rect.x + 14, self.card_rect.bottom - 36, 220, 26)

        self.slot_labels = []
        self.slot_rects = []
        self.dock_rect = pygame.Rect(0, 0, 0, 0)
        self.hint_surf = None
        self.set_lang(lang)

    def set_lang(self, lang):
        self.lang = lang
        self._surface_cache.clear()
        self.dock_rect = self._build_dock()
        t = get_text(self.lang)
        self.hint_surf = self.small.render(t["hud_hint"], True, (148, 162, 184))
        self.hint_surf.set_alpha(175)

    def _build_dock(self):
        self.slot_labels = []
        self.slot_rects = []
        gap = 6
        widths = []
        for spec in SPAWN_CATALOG:
            name = body_label(self.lang, spec["id"], spec["name"])
            active_s = self.bold.render(name, True, (245, 250, 255))
            idle_s = self.font.render(name, True, (188, 198, 214))
            self.slot_labels.append((name, active_s, idle_s))
            widths.append(active_s.get_width() + 34)

        total_w = sum(widths) + gap * (len(widths) - 1)
        pad_x = 10
        dock_w = total_w + pad_x * 2
        dock_x = (W - dock_w) // 2
        dock_y = H - 64

        x = dock_x + pad_x
        for w in widths:
            self.slot_rects.append(pygame.Rect(x, dock_y + 6, w, 30))
            x += w + gap

        return pygame.Rect(dock_x, dock_y, dock_w, 42)

    def _glass(self, size, fill=(8, 12, 20, 182), border=(185, 205, 232, 72), radius=10):
        key = (size, fill, border, radius)
        surf = self._surface_cache.get(key)
        if surf is None:
            surf = pygame.Surface(size, pygame.SRCALPHA)
            r = surf.get_rect()
            pygame.draw.rect(surf, fill, r, border_radius=radius)
            if border is not None:
                pygame.draw.rect(surf, border, r, 1, border_radius=radius)
            if len(self._surface_cache) > 96:
                self._surface_cache.clear()
            self._surface_cache[key] = surf
        return surf

    def notify(self, msg, dur=1.8):
        self.toast = msg
        self.toast_end = time.time() + dur

    def is_over_ui(self, pos, show_hud, has_card=False):
        if not show_hud:
            return False
        if (
            self.menu_btn.collidepoint(pos)
            or self.status_rect.collidepoint(pos)
            or self.time_panel.collidepoint(pos)
            or self.dock_rect.collidepoint(pos)
        ):
            return True
        if has_card and self.card_rect.collidepoint(pos):
            return True
        return False

    def handle_click(self, pos, show_hud, has_focus=False):
        if not show_hud:
            return None
        if self.menu_btn.collidepoint(pos):
            return ("menu", None)
        if self.btn_slow.collidepoint(pos):
            return ("speed_down", None)
        if self.btn_fast.collidepoint(pos):
            return ("speed_up", None)
        if self.btn_pause.collidepoint(pos):
            return ("pause", None)
        if has_focus and self.btn_unlock.collidepoint(pos):
            return ("unlock", None)
        for i, r in enumerate(self.slot_rects):
            if r.collidepoint(pos):
                return ("slot", i)
        return None

    def draw_slingshot(self, screen, start_px, cur_px, spawn_spec):
        t = get_text(self.lang)
        pygame.draw.circle(screen, spawn_spec["atmo"], start_px, 8, 2)
        pygame.draw.line(screen, (235, 242, 255), start_px, cur_px, 2)
        pygame.draw.circle(screen, (245, 248, 255), cur_px, 4)

        name = body_label(self.lang, spawn_spec["id"], spawn_spec["name"])
        lbl = self.small.render(t["slingshot"].format(name=name), True, (240, 246, 255))
        box = pygame.Rect(start_px[0] + 12, start_px[1] - 26, lbl.get_width() + 16, 22)
        screen.blit(self._glass(box.size, (8, 12, 20, 200), (210, 226, 248, 120), 8), box.topleft)
        screen.blit(lbl, (box.x + 8, box.y + 3))

    def draw_reticle(self, screen, body, cam_pos, fwd, right, up, locked=False):
        if body is None:
            return
        proj = world_to_screen(body["pos"], cam_pos, fwd, right, up)
        if proj is None:
            return
        sx, sy, z = proj
        rad = max(16, min(135, int(body["radius"] * 1.75 * (FOCAL / z)))) + 8
        if sx < -rad or sx > W + rad or sy < -rad or sy > H + rad:
            return

        col = (210, 236, 255) if locked else (165, 185, 210)
        arm = max(6, min(16, rad // 3))
        x0, x1 = sx - rad, sx + rad
        y0, y1 = sy - rad, sy + rad

        pygame.draw.line(screen, col, (x0, y0), (x0 + arm, y0), 1)
        pygame.draw.line(screen, col, (x0, y0), (x0, y0 + arm), 1)

        pygame.draw.line(screen, col, (x1, y0), (x1 - arm, y0), 1)
        pygame.draw.line(screen, col, (x1, y0), (x1, y0 + arm), 1)

        pygame.draw.line(screen, col, (x0, y1), (x0 + arm, y1), 1)
        pygame.draw.line(screen, col, (x0, y1), (x0, y1 - arm), 1)

        pygame.draw.line(screen, col, (x1, y1), (x1 - arm, y1), 1)
        pygame.draw.line(screen, col, (x1, y1), (x1, y1 - arm), 1)

    def _draw_card(self, screen, body, locked, mp):
        t = get_text(self.lang)
        is_bh = body["kind"] == "bh"
        card_h = 196 if is_bh else 174
        self.card_rect.height = card_h
        self.btn_unlock.y = self.card_rect.bottom - 36

        screen.blit(self._glass(self.card_rect.size, radius=10), self.card_rect.topleft)

        name = body_label(self.lang, body["id"], body["name"])
        kind_lbl = t["kinds"].get(body["kind"], body["kind"])

        cx = self.card_rect.x + 14
        cy = self.card_rect.y + 14
        pygame.draw.circle(screen, body["color"], (cx + 7, cy + 10), 7)
        pygame.draw.circle(screen, body["atmo"], (cx + 7, cy + 10), 7, 1)

        screen.blit(self.title_font.render(name, True, (242, 247, 255)), (cx + 22, cy - 1))
        screen.blit(self.small.render(kind_lbl, True, (150, 166, 188)), (cx + 22, cy + 20))

        div_y = cy + 42
        pygame.draw.line(screen, (68, 82, 104), (cx, div_y), (self.card_rect.right - 14, div_y), 1)

        speed = float(np.linalg.norm(body["vel"]))
        rows = [
            (t["card_mass"], f"{body['mass']:.1f}"),
            (t["card_speed"], f"{speed:.2f} {t['card_speed_unit']}"),
            (t["card_radius"], f"{body['radius']:.2f}"),
        ]
        if is_bh:
            rows.append((t["card_absorbed"], f"{body['absorbed']:,} {t['card_absorbed_unit']}"))

        ry = div_y + 10
        for label, val in rows:
            screen.blit(self.small.render(label, True, (152, 168, 190)), (cx, ry))
            vs = self.meta.render(val, True, (232, 240, 252))
            screen.blit(vs, (self.card_rect.right - 14 - vs.get_width(), ry))
            ry += 21

        if locked:
            hover = self.btn_unlock.collidepoint(mp)
            fill = (215, 232, 255, 46 if hover else 24)
            bord = (230, 242, 255, 210 if hover else 135)
            screen.blit(self._glass(self.btn_unlock.size, fill, bord, 13), self.btn_unlock.topleft)
            lbl = self.small.render(t["btn_unlock"], True, (242, 248, 255))
            screen.blit(
                lbl,
                (
                    self.btn_unlock.centerx - lbl.get_width() // 2,
                    self.btn_unlock.centery - lbl.get_height() // 2,
                ),
            )
        else:
            hint = self.small.render(t["card_hint_lock"], True, (142, 162, 188))
            screen.blit(hint, (self.card_rect.centerx - hint.get_width() // 2, self.card_rect.bottom - 26))

    def draw(
        self,
        screen,
        preset_name,
        n_stars,
        bodies,
        ascii_mode,
        paused,
        time_scale,
        fps,
        selected_idx,
        focused_body=None,
        hovered_body=None,
    ):
        t = get_text(self.lang)
        mp = pygame.mouse.get_pos()

        m_hover = self.menu_btn.collidepoint(mp)
        screen.blit(
            self._glass(
                self.menu_btn.size,
                (215, 232, 255, 40) if m_hover else (8, 12, 20, 182),
                (230, 242, 255, 190) if m_hover else (185, 205, 232, 72),
                16,
            ),
            self.menu_btn.topleft,
        )
        m_txt = self.bold.render(t["hud_menu"], True, (245, 250, 255) if m_hover else (216, 226, 242))
        screen.blit(m_txt, (self.menu_btn.centerx - m_txt.get_width() // 2, self.menu_btn.centery - m_txt.get_height() // 2))

        p_loc = preset_label(self.lang, preset_name)
        body_cnt = sum(1 for b in bodies if b["kind"] != "core")
        mode_s = "ASCII" if ascii_mode else "HD"
        status_txt = f"{p_loc}   ·   {n_stars // 1000}k {t['hud_stars_unit']}   ·   {t['hud_bodies']}: {body_cnt}   ·   {mode_s}"
        s_surf = self.font.render(status_txt, True, (224, 232, 244))
        self.status_rect.width = s_surf.get_width() + 28
        screen.blit(self._glass(self.status_rect.size, radius=16), self.status_rect.topleft)
        screen.blit(s_surf, (self.status_rect.x + 14, self.status_rect.centery - s_surf.get_height() // 2))

        screen.blit(self._glass(self.time_panel.size, radius=16), self.time_panel.topleft)

        for btn, sym in ((self.btn_slow, "−"), (self.btn_fast, "+")):
            hov = btn.collidepoint(mp)
            screen.blit(
                self._glass(
                    btn.size,
                    (215, 232, 255, 44 if hov else 18),
                    (230, 242, 255, 185 if hov else 85),
                    12,
                ),
                btn.topleft,
            )
            st = self.bold.render(sym, True, (245, 250, 255))
            screen.blit(st, (btn.centerx - st.get_width() // 2, btn.centery - st.get_height() // 2 - 1))

        spd_str = f"{time_scale:g}x"
        spd_col = (245, 248, 255) if not paused else (142, 154, 174)
        spd_s = self.meta.render(spd_str, True, spd_col)
        screen.blit(spd_s, (self.btn_slow.right + (self.btn_fast.left - self.btn_slow.right - spd_s.get_width()) // 2, self.time_panel.centery - spd_s.get_height() // 2))

        p_hov = self.btn_pause.collidepoint(mp)
        p_fill = (255, 198, 120, 52 if p_hov else 30) if paused else (215, 232, 255, 44 if p_hov else 18)
        p_bord = (255, 212, 145, 200) if paused else (230, 242, 255, 185 if p_hov else 85)
        screen.blit(self._glass(self.btn_pause.size, p_fill, p_bord, 12), self.btn_pause.topleft)
        p_lbl = self.small.render(t["hud_play"] if paused else t["hud_pause"], True, (255, 232, 190) if paused else (228, 236, 248))
        screen.blit(p_lbl, (self.btn_pause.centerx - p_lbl.get_width() // 2, self.btn_pause.centery - p_lbl.get_height() // 2))

        fps_s = self.meta.render(f"{fps:2d} FPS", True, (176, 188, 206))
        screen.blit(fps_s, (self.time_panel.right - fps_s.get_width() - 12, self.time_panel.centery - fps_s.get_height() // 2))

        card_body = focused_body if focused_body is not None else hovered_body
        if card_body is not None:
            self._draw_card(screen, card_body, focused_body is not None, mp)

        screen.blit(self._glass(self.dock_rect.size, (8, 12, 20, 188), (185, 205, 232, 72), 21), self.dock_rect.topleft)
        for i, (spec, rect) in enumerate(zip(SPAWN_CATALOG, self.slot_rects)):
            active = i == selected_idx
            hover = rect.collidepoint(mp)
            if active:
                screen.blit(
                    self._glass(rect.size, (215, 232, 255, 44), (230, 242, 255, 205), 15),
                    rect.topleft,
                )
            elif hover:
                screen.blit(
                    self._glass(rect.size, (215, 232, 255, 22), (215, 232, 255, 110), 15),
                    rect.topleft,
                )
            pygame.draw.circle(screen, spec["color"], (rect.x + 13, rect.centery), 5)
            pygame.draw.circle(screen, spec["atmo"], (rect.x + 13, rect.centery), 5, 1)
            _, active_s, idle_s = self.slot_labels[i]
            txt_s = active_s if (active or hover) else idle_s
            screen.blit(txt_s, (rect.x + 24, rect.centery - txt_s.get_height() // 2))

        screen.blit(self.hint_surf, (W // 2 - self.hint_surf.get_width() // 2, H - 18))

        rem = self.toast_end - time.time()
        if rem > 0.0 and self.toast:
            alpha = min(255, int(min(1.0, rem / 0.4) * 255))
            ts = self.font.render(self.toast, True, (242, 247, 255))
            box = pygame.Rect(18, 56, ts.get_width() + 24, 28)
            pill = self._glass(box.size, (8, 12, 20, 198), (210, 228, 252, 130), 14).copy()
            pill.set_alpha(alpha)
            ts.set_alpha(alpha)
            screen.blit(pill, box.topleft)
            screen.blit(ts, (box.x + 12, box.centery - ts.get_height() // 2))
