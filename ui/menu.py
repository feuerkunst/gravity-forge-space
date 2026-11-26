import time
import pygame
from config import W, H, STAR_OPTIONS
from sim.presets import PRESET_INFO
from ui.lang import get_text


class MainMenu:
    def __init__(self):
        self.title_font = pygame.font.SysFont("bahnschrift,segoeui,arial", 40, bold=True)
        self.menu_font = pygame.font.SysFont("segoeui,arial,tahoma", 17, bold=True)
        self.head_font = pygame.font.SysFont("segoeui,arial,tahoma", 16, bold=True)
        self.item_font = pygame.font.SysFont("segoeui,arial,tahoma", 15)
        self.desc_font = pygame.font.SysFont("segoeui,arial,tahoma", 12)
        self.meta_font = pygame.font.SysFont("consolas,segoeui,arial", 12)

        self.vignette = self._build_vignette()

        self.menu_keys = ("continue", "planetarium", "settings", "editor", "quit")
        self.item_rects = []
        y = 282
        for key in self.menu_keys:
            if key == "quit":
                y += 28
            self.item_rects.append((key, pygame.Rect(82, y, 270, 38)))
            y += 46

        self.active_panel = None
        self.panel_rect = pygame.Rect(450, 136, 690, 480)
        self.close_rect = pygame.Rect(self.panel_rect.right - 38, self.panel_rect.y + 14, 24, 24)
        self.preset_rects = [
            pygame.Rect(self.panel_rect.x + 28, self.panel_rect.y + 66 + i * 64, 634, 54)
            for i in range(len(PRESET_INFO))
        ]
        self.launch_panel_rect = pygame.Rect(self.panel_rect.x + 28, self.panel_rect.bottom - 56, 250, 38)
        self.setting_rects = [
            pygame.Rect(self.panel_rect.x + 28, self.panel_rect.y + 62 + i * 44, 634, 36)
            for i in range(6)
        ]

        self._last_t = time.perf_counter()
        self._fps = 60.0

    def _build_vignette(self):
        vw = 520
        surf = pygame.Surface((vw, H), pygame.SRCALPHA)
        for x in range(vw):
            t = 1.0 - (x / vw)
            alpha = int((t ** 1.7) * 145)
            pygame.draw.line(surf, (4, 6, 12, alpha), (x, 0), (x, H))
        return surf

    def _draw_text(self, screen, font, text, pos, color):
        sh = font.render(text, True, (0, 0, 0))
        sh.set_alpha(175)
        screen.blit(sh, (pos[0] + 1, pos[1] + 2))
        screen.blit(font.render(text, True, color), pos)

    def _draw_pill(self, screen, rect, fill_alpha=34, border_alpha=185):
        pill = pygame.Surface(rect.size, pygame.SRCALPHA)
        r = rect.height // 2
        pygame.draw.rect(pill, (215, 232, 255, fill_alpha), pill.get_rect(), border_radius=r)
        pygame.draw.rect(pill, (230, 242, 255, border_alpha), pill.get_rect(), 1, border_radius=r)
        screen.blit(pill, rect.topleft)

    def handle_event(self, ev, cfg, has_session):
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_RETURN:
                self.active_panel = None
                return "launch"
            if ev.key == pygame.K_ESCAPE:
                if self.active_panel is not None:
                    self.active_panel = None
                    return None
                return "resume" if has_session else "quit"
            if ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                cfg["preset"] = ev.key - pygame.K_0
                return "preset_preview"

        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            mp = ev.pos

            for key, r in self.item_rects:
                if r.collidepoint(mp):
                    if key == "continue":
                        self.active_panel = None
                        return "resume" if has_session else "launch"
                    if key == "planetarium":
                        self.active_panel = None
                        return "launch"
                    if key == "settings":
                        self.active_panel = None if self.active_panel == "settings" else "settings"
                        return None
                    if key == "editor":
                        self.active_panel = None if self.active_panel == "editor" else "editor"
                        return None
                    if key == "quit":
                        return "quit"

            if self.active_panel is not None:
                if self.close_rect.collidepoint(mp):
                    self.active_panel = None
                    return None

                if self.active_panel == "editor":
                    for idx, r in enumerate(self.preset_rects):
                        if r.collidepoint(mp):
                            cfg["preset"] = PRESET_INFO[idx][0]
                            return "preset_preview"
                    if self.launch_panel_rect.collidepoint(mp):
                        self.active_panel = None
                        return "launch"

                elif self.active_panel == "settings":
                    if self.setting_rects[0].collidepoint(mp):
                        cfg["lang"] = "en" if cfg.get("lang", "ru") == "ru" else "ru"
                        return "lang_changed"
                    if self.setting_rects[1].collidepoint(mp):
                        cur = STAR_OPTIONS.index(cfg["stars"]) if cfg["stars"] in STAR_OPTIONS else 1
                        cfg["stars"] = STAR_OPTIONS[(cur + 1) % len(STAR_OPTIONS)]
                        return "setting_changed"
                    if self.setting_rects[2].collidepoint(mp):
                        cfg["ascii"] = not cfg["ascii"]
                        return "setting_changed"
                    if self.setting_rects[3].collidepoint(mp):
                        cfg["trails"] = not cfg["trails"]
                        return "setting_changed"
                    if self.setting_rects[4].collidepoint(mp):
                        cfg["orbits"] = not cfg["orbits"]
                        return "setting_changed"
                    if self.setting_rects[5].collidepoint(mp):
                        cfg["jets"] = not cfg["jets"]
                        return "setting_changed"

                if not self.panel_rect.collidepoint(mp):
                    self.active_panel = None

        return None

    def _draw_side_panel(self, screen, cfg, mp):
        t = get_text(cfg.get("lang", "ru"))

        panel = pygame.Surface(self.panel_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(panel, (8, 12, 20, 188), panel.get_rect(), border_radius=10)
        pygame.draw.rect(panel, (185, 205, 232, 72), panel.get_rect(), 1, border_radius=10)
        screen.blit(panel, self.panel_rect.topleft)

        title = t["panel_settings"] if self.active_panel == "settings" else t["panel_editor"]
        self._draw_text(screen, self.head_font, title, (self.panel_rect.x + 28, self.panel_rect.y + 20), (236, 242, 252))
        pygame.draw.line(
            screen,
            (90, 108, 134),
            (self.panel_rect.x + 28, self.panel_rect.y + 48),
            (self.panel_rect.right - 28, self.panel_rect.y + 48),
            1,
        )

        c_col = (245, 248, 255) if self.close_rect.collidepoint(mp) else (150, 165, 185)
        cx, cy = self.close_rect.center
        pygame.draw.line(screen, c_col, (cx - 5, cy - 5), (cx + 5, cy + 5), 2)
        pygame.draw.line(screen, c_col, (cx + 5, cy - 5), (cx - 5, cy + 5), 2)

        if self.active_panel == "editor":
            for (pid, fallback_name, fallback_desc), r in zip(PRESET_INFO, self.preset_rects):
                name, desc = t["scenarios"].get(pid, (fallback_name, fallback_desc))
                selected = cfg["preset"] == pid
                hover = r.collidepoint(mp)

                row = pygame.Surface(r.size, pygame.SRCALPHA)
                if selected:
                    pygame.draw.rect(row, (215, 232, 255, 38), row.get_rect(), border_radius=7)
                    pygame.draw.rect(row, (230, 242, 255, 185), row.get_rect(), 1, border_radius=7)
                elif hover:
                    pygame.draw.rect(row, (215, 232, 255, 20), row.get_rect(), border_radius=7)
                    pygame.draw.rect(row, (210, 226, 248, 95), row.get_rect(), 1, border_radius=7)
                else:
                    pygame.draw.rect(row, (18, 24, 36, 110), row.get_rect(), border_radius=7)
                screen.blit(row, r.topleft)

                self._draw_text(screen, self.item_font, f"{pid}. {name}", (r.x + 14, r.y + 8), (242, 246, 255))
                screen.blit(self.desc_font.render(desc, True, (162, 176, 198)), (r.x + 14, r.y + 30))

            l_hover = self.launch_panel_rect.collidepoint(mp)
            self._draw_pill(
                screen,
                self.launch_panel_rect,
                fill_alpha=52 if l_hover else 28,
                border_alpha=225 if l_hover else 140,
            )
            lbl = self.menu_font.render(t["btn_launch"], True, (245, 250, 255))
            screen.blit(
                lbl,
                (
                    self.launch_panel_rect.centerx - lbl.get_width() // 2,
                    self.launch_panel_rect.centery - lbl.get_height() // 2,
                ),
            )

        elif self.active_panel == "settings":
            rows = [
                (t["set_lang"], t["lang_name"], True),
                (t["set_stars"], f"{cfg['stars']:,}", True),
                (t["set_render"], "ASCII" if cfg["ascii"] else "HD", True),
                (t["set_trails"], t["on"] if cfg["trails"] else t["off"], cfg["trails"]),
                (t["set_orbits"], t["on"] if cfg["orbits"] else t["off"], cfg["orbits"]),
                (t["set_jets"], t["on"] if cfg["jets"] else t["off"], cfg["jets"]),
            ]
            for (label, val, is_on), r in zip(rows, self.setting_rects):
                hover = r.collidepoint(mp)
                row = pygame.Surface(r.size, pygame.SRCALPHA)
                pygame.draw.rect(
                    row,
                    (215, 232, 255, 28 if hover else 12),
                    row.get_rect(),
                    border_radius=6,
                )
                if hover:
                    pygame.draw.rect(row, (220, 235, 255, 130), row.get_rect(), 1, border_radius=6)
                screen.blit(row, r.topleft)

                self._draw_text(screen, self.item_font, label, (r.x + 14, r.y + 8), (225, 234, 246))
                val_col = (242, 248, 255) if is_on else (150, 162, 180)
                vs = self.item_font.render(val, True, val_col)
                screen.blit(vs, (r.right - vs.get_width() - 16, r.y + 8))

            hy = self.panel_rect.y + 338
            for line in t["menu_hints"]:
                screen.blit(self.desc_font.render(line, True, (152, 168, 190)), (self.panel_rect.x + 32, hy))
                hy += 21

    def draw(self, screen, cfg, has_session):
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        if 0.0005 < dt < 0.5:
            self._fps = self._fps * 0.9 + (1.0 / dt) * 0.1

        t = get_text(cfg.get("lang", "ru"))
        mp = pygame.mouse.get_pos()
        screen.blit(self.vignette, (0, 0))

        self._draw_text(screen, self.title_font, "GRAVITY FORGE", (128, 92), (245, 248, 255))

        for (key, label), (_, r) in zip(t["menu_items"], self.item_rects):
            hover = r.collidepoint(mp)
            is_open = (key == "settings" and self.active_panel == "settings") or (
                key == "editor" and self.active_panel == "editor"
            )
            if hover or is_open:
                self._draw_pill(
                    screen,
                    r,
                    fill_alpha=40 if hover else 26,
                    border_alpha=205 if hover else 145,
                )

            if key == "continue" and not has_session:
                col = (210, 218, 232) if hover else (150, 158, 172)
            else:
                col = (255, 255, 255) if (hover or is_open) else (228, 234, 244)

            self._draw_text(screen, self.menu_font, label, (r.x + 48, r.y + 8), col)

        if self.active_panel is not None:
            self._draw_side_panel(screen, cfg, mp)

        ver = self.meta_font.render("1 | 0   ALPHA", True, (168, 178, 194))
        ver.set_alpha(165)
        screen.blit(ver, (128, H - 36))

        fps_txt = self.meta_font.render(f"FPS {int(round(self._fps))}", True, (186, 196, 212))
        fps_txt.set_alpha(180)
        screen.blit(fps_txt, (W - fps_txt.get_width() - 22, 16))
