import ctypes
import time
import numpy as np
import pygame

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

from config import W, H, DEFAULT_STARS
from sim.bodies import SPAWN_CATALOG, make_body, compute_orbit_velocity
from sim.presets import load_preset, make_background_stars
from sim.physics import step_physics
from render.camera import get_camera, screen_to_world, pick_body
from render.hd import HDRenderer
from render.ascii import AsciiRenderer
from render.planets import PlanetRenderer
from ui.menu import MainMenu
from ui.hud import HUD
from ui.lang import get_text, body_label, preset_label

SPEED_STEPS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_SPEED_IDX = 3


def main():
    pygame.init()
    pygame.display.set_caption("Gravity Forge")
    screen = pygame.display.set_mode((W, H), pygame.DOUBLEBUF)
    clock = pygame.time.Clock()

    cfg = {
        "lang": "ru",
        "preset": 1,
        "stars": DEFAULT_STARS,
        "ascii": False,
        "trails": True,
        "orbits": True,
        "jets": True,
    }

    hd_renderer = HDRenderer()
    ascii_renderer = AsciiRenderer()
    planet_renderer = PlanetRenderer()
    menu = MainMenu()
    hud = HUD(lang=cfg["lang"])
    bg_stars = make_background_stars()

    pos, vel, base_cols, bodies, preset_name = load_preset(cfg["preset"], cfg["stars"])

    yaw, pitch, dist = 0.25, 0.42, 38.0
    target = np.zeros(3, dtype=np.float32)

    in_menu = True
    has_session = False
    paused = False
    speed_idx = DEFAULT_SPEED_IDX
    show_hud = True
    fullscreen = False
    auto_spin = True

    focused_body = None
    hovered_body = None

    dragging_cam = False
    lmb_moved = False
    lmb_start = (0, 0)
    last_m = (0, 0)
    selected_spawn = 0

    aiming_spawn = False
    aim_start_px = (0, 0)
    aim_world = np.zeros(3, dtype=np.float32)

    def tr(key, **kw):
        msg = get_text(cfg["lang"])[key]
        return msg.format(**kw) if kw else msg

    def set_speed(new_idx):
        nonlocal speed_idx
        new_idx = max(0, min(len(SPEED_STEPS) - 1, new_idx))
        if new_idx != speed_idx:
            speed_idx = new_idx
            hud.notify(tr("t_speed", v=f"{SPEED_STEPS[speed_idx]:g}"))

    def cycle_focus(step):
        nonlocal focused_body, auto_spin
        pool = [b for b in bodies if b["kind"] != "core"]
        if not pool:
            hud.notify(tr("t_no_focus"))
            return
        if focused_body in pool:
            idx = (pool.index(focused_body) + step) % len(pool)
        else:
            idx = 0 if step > 0 else len(pool) - 1
        focused_body = pool[idx]
        auto_spin = False
        hud.notify(tr("t_cam_focus", name=body_label(cfg["lang"], focused_body["id"], focused_body["name"])))

    def start_simulation():
        nonlocal pos, vel, base_cols, bodies, preset_name
        nonlocal yaw, pitch, dist, target, paused, speed_idx, auto_spin, in_menu, has_session
        nonlocal focused_body, hovered_body
        pos, vel, base_cols, bodies, preset_name = load_preset(cfg["preset"], cfg["stars"])
        hd_renderer.clear_trails()
        yaw, pitch, dist = 0.25, 0.42, 38.0 if cfg["preset"] != 5 else 28.0
        target = np.zeros(3, dtype=np.float32)
        paused = False
        speed_idx = DEFAULT_SPEED_IDX
        auto_spin = True
        focused_body = None
        hovered_body = None
        in_menu = False
        has_session = True
        hud.notify(tr("t_scenario", name=preset_label(cfg["lang"], preset_name)))

    running = True
    while running:
        real_dt = min(clock.tick(60) / 1000.0, 0.04)
        time_scale = SPEED_STEPS[speed_idx]
        has_card = (focused_body is not None) or (hovered_body is not None)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_F11:
                fullscreen = not fullscreen
                fl = (pygame.FULLSCREEN | pygame.SCALED | pygame.DOUBLEBUF) if fullscreen else pygame.DOUBLEBUF
                screen = pygame.display.set_mode((W, H), fl)
                continue

            if in_menu:
                action = menu.handle_event(ev, cfg, has_session)
                if action == "launch":
                    start_simulation()
                elif action == "resume" and has_session:
                    in_menu = False
                elif action == "quit":
                    running = False
                elif action == "lang_changed":
                    hud.set_lang(cfg["lang"])
                elif action == "preset_preview" and not has_session:
                    pos, vel, base_cols, bodies, preset_name = load_preset(cfg["preset"], cfg["stars"])
                    hd_renderer.clear_trails()
                    target = np.zeros(3, dtype=np.float32)
                elif action == "setting_changed" and not has_session:
                    if pos.shape[0] != cfg["stars"]:
                        pos, vel, base_cols, bodies, preset_name = load_preset(cfg["preset"], cfg["stars"])
                        hd_renderer.clear_trails()
                continue

            if ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE:
                    in_menu = True
                    dragging_cam = False
                    aiming_spawn = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                    hud.notify(tr("t_paused") if paused else tr("t_resumed", v=f"{time_scale:g}"))
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    set_speed(speed_idx - 1)
                elif k in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    set_speed(speed_idx + 1)
                elif k == pygame.K_t:
                    set_speed(1 if speed_idx != 1 else DEFAULT_SPEED_IDX)
                elif k == pygame.K_LEFTBRACKET:
                    cycle_focus(-1)
                elif k == pygame.K_RIGHTBRACKET:
                    cycle_focus(1)
                elif k == pygame.K_f:
                    if focused_body is not None:
                        focused_body = None
                        hud.notify(tr("t_cam_unlocked"))
                elif k == pygame.K_TAB:
                    cfg["ascii"] = not cfg["ascii"]
                    hd_renderer.clear_trails()
                    hud.notify(tr("t_mode", mode="ASCII" if cfg["ascii"] else "HD"))
                elif k == pygame.K_g:
                    cfg["trails"] = not cfg["trails"]
                    hd_renderer.clear_trails()
                    hud.notify(tr("t_trails_on" if cfg["trails"] else "t_trails_off"))
                elif k == pygame.K_o:
                    cfg["orbits"] = not cfg["orbits"]
                    hud.notify(tr("t_orbits_on" if cfg["orbits"] else "t_orbits_off"))
                elif k == pygame.K_j:
                    cfg["jets"] = not cfg["jets"]
                    hud.notify(tr("t_jets_on" if cfg["jets"] else "t_jets_off"))
                elif k == pygame.K_h:
                    show_hud = not show_hud
                elif k == pygame.K_v:
                    focused_body = None
                    yaw, pitch, dist = 0.25, 0.42, 38.0 if cfg["preset"] != 5 else 28.0
                    auto_spin = True
                    hud.notify(tr("t_cam_reset"))
                elif k == pygame.K_q:
                    selected_spawn = (selected_spawn - 1) % len(SPAWN_CATALOG)
                    spec = SPAWN_CATALOG[selected_spawn]
                    hud.notify(tr("t_object", name=body_label(cfg["lang"], spec["id"], spec["name"])))
                elif k == pygame.K_e:
                    selected_spawn = (selected_spawn + 1) % len(SPAWN_CATALOG)
                    spec = SPAWN_CATALOG[selected_spawn]
                    hud.notify(tr("t_object", name=body_label(cfg["lang"], spec["id"], spec["name"])))
                elif k == pygame.K_r:
                    start_simulation()
                elif k == pygame.K_c:
                    bodies = [b for b in bodies if b["kind"] == "core"]
                    focused_body = None
                    hovered_body = None
                    hud.notify(tr("t_cleared"))
                elif k in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                    cfg["preset"] = k - pygame.K_0
                    start_simulation()
                elif k == pygame.K_s:
                    fname = f"gravity_{int(time.time())}.png"
                    pygame.image.save(screen, fname)
                    hud.notify(tr("t_screenshot", fname=fname), 2.2)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    hit_ui = hud.handle_click(ev.pos, show_hud, focused_body is not None)
                    if hit_ui is not None:
                        act, val = hit_ui
                        if act == "menu":
                            in_menu = True
                            dragging_cam = False
                            aiming_spawn = False
                        elif act == "speed_down":
                            set_speed(speed_idx - 1)
                        elif act == "speed_up":
                            set_speed(speed_idx + 1)
                        elif act == "pause":
                            paused = not paused
                        elif act == "unlock":
                            focused_body = None
                            hud.notify(tr("t_cam_unlocked"))
                        elif act == "slot":
                            selected_spawn = val
                            spec = SPAWN_CATALOG[selected_spawn]
                            hud.notify(tr("t_selected", name=body_label(cfg["lang"], spec["id"], spec["name"])))
                        continue

                    if not hud.is_over_ui(ev.pos, show_hud, has_card):
                        dragging_cam = True
                        lmb_moved = False
                        lmb_start = ev.pos
                        last_m = ev.pos

                elif ev.button == 3 and not hud.is_over_ui(ev.pos, show_hud, has_card):
                    _, _, right, up = get_camera(yaw, pitch, dist, target)
                    aim_world = screen_to_world(ev.pos[0], ev.pos[1], target, right, up, dist)
                    aim_start_px = ev.pos
                    aiming_spawn = True

            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1 and dragging_cam:
                    dragging_cam = False
                    if not lmb_moved:
                        cam_p, fwd_v, right_v, up_v = get_camera(yaw, pitch, dist, target)
                        picked = pick_body(ev.pos[0], ev.pos[1], bodies, cam_p, fwd_v, right_v, up_v)
                        if picked is not None:
                            focused_body = picked
                            auto_spin = False
                            hud.notify(tr("t_cam_focus", name=body_label(cfg["lang"], picked["id"], picked["name"])))

                elif ev.button == 3 and aiming_spawn:
                    aiming_spawn = False
                    dx_px = ev.pos[0] - aim_start_px[0]
                    dy_px = ev.pos[1] - aim_start_px[1]
                    spec = SPAWN_CATALOG[selected_spawn]
                    loc_name = body_label(cfg["lang"], spec["id"], spec["name"])
                    mods = pygame.key.get_mods()

                    if dx_px * dx_px + dy_px * dy_px > 100:
                        _, _, right, up = get_camera(yaw, pitch, dist, target)
                        launch_vel = (right * dx_px - up * dy_px) * 0.045
                        bodies.append(make_body(spec["id"], aim_world, launch_vel))
                        hud.notify(tr("t_launched", name=loc_name))
                    elif (mods & pygame.KMOD_SHIFT) or spec["kind"] == "repulsor":
                        bodies.append(make_body(spec["id"], aim_world, np.zeros(3, dtype=np.float32)))
                        hud.notify(tr("t_spawned", name=loc_name))
                    else:
                        orb_v = compute_orbit_velocity(aim_world, bodies)
                        bodies.append(make_body(spec["id"], aim_world, orb_v))
                        hud.notify(tr("t_orbiting", name=loc_name))

            elif ev.type == pygame.MOUSEMOTION and dragging_cam:
                dx = ev.pos[0] - last_m[0]
                dy = ev.pos[1] - last_m[1]
                last_m = ev.pos
                if not lmb_moved:
                    tdx = ev.pos[0] - lmb_start[0]
                    tdy = ev.pos[1] - lmb_start[1]
                    if tdx * tdx + tdy * tdy > 25:
                        lmb_moved = True
                        auto_spin = False
                if lmb_moved:
                    yaw += dx * 0.0055
                    pitch = max(-1.45, min(1.45, pitch + dy * 0.0055))

            elif ev.type == pygame.MOUSEWHEEL:
                dist = max(3.2, min(140.0, dist * (0.91 if ev.y > 0 else 1.1)))

        if in_menu:
            yaw += real_dt * 0.14
            if not has_session:
                step_physics(pos, vel, bodies, real_dt * 0.35, jets_enabled=cfg["jets"])
        else:
            if auto_spin and not paused:
                yaw += real_dt * 0.11

            prev_focus_pos = focused_body["pos"].copy() if (focused_body is not None and focused_body in bodies) else None

            if not paused:
                sim_dt = real_dt * time_scale
                substeps = 2 if time_scale <= 1.0 else (3 if time_scale <= 2.0 else 4)
                sub_dt = sim_dt / substeps
                for _ in range(substeps):
                    step_physics(pos, vel, bodies, sub_dt, jets_enabled=cfg["jets"])

            if focused_body is not None and focused_body not in bodies:
                focused_body = None
                hud.notify(tr("t_absorbed"))
            elif focused_body is not None and prev_focus_pos is not None:
                target += focused_body["pos"] - prev_focus_pos

        if focused_body is not None:
            target += (focused_body["pos"] - target) * min(1.0, real_dt * 14.0)
        else:
            anchors = [b for b in bodies if b["kind"] in ("core", "star")]
            if anchors:
                total_m = sum(b["mass"] for b in anchors)
                if total_m > 0:
                    com = sum(b["pos"] * b["mass"] for b in anchors) / total_m
                    target += (com - target) * min(1.0, real_dt * 4.0)

        cam_pos, fwd, right, up = get_camera(yaw, pitch, dist, target)

        if not in_menu and not dragging_cam and not aiming_spawn:
            mp = pygame.mouse.get_pos()
            if not hud.is_over_ui(mp, show_hud, has_card):
                hovered_body = pick_body(mp[0], mp[1], bodies, cam_pos, fwd, right, up)
            else:
                hovered_body = None
        else:
            hovered_body = None

        if cfg["ascii"]:
            ascii_renderer.render(screen, pos, base_cols, cam_pos, fwd, right, up)
        else:
            hd_renderer.render(screen, pos, vel, base_cols, cam_pos, fwd, right, up, bg_stars, cfg["trails"])

        planet_renderer.draw_bodies(screen, bodies, cam_pos, fwd, right, up, show_orbits=cfg["orbits"])

        if not in_menu:
            if show_hud:
                if focused_body is not None:
                    hud.draw_reticle(screen, focused_body, cam_pos, fwd, right, up, locked=True)
                elif hovered_body is not None:
                    hud.draw_reticle(screen, hovered_body, cam_pos, fwd, right, up, locked=False)

            if aiming_spawn:
                hud.draw_slingshot(screen, aim_start_px, pygame.mouse.get_pos(), SPAWN_CATALOG[selected_spawn])

            if show_hud:
                hud.draw(
                    screen,
                    preset_name,
                    pos.shape[0],
                    bodies,
                    cfg["ascii"],
                    paused,
                    time_scale,
                    int(clock.get_fps()),
                    selected_spawn,
                    focused_body=focused_body,
                    hovered_body=hovered_body,
                )
        else:
            menu.draw(screen, cfg, has_session)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
