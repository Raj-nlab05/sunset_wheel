import sys
import math
import random
import time

import cv2
import pygame
import mediapipe as mp

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

WIN_W, WIN_H = 1280, 720
FPS = 60

N_SLICES = 8
CENTER_X, CENTER_Y = WIN_W // 2, WIN_H // 2
OUTER_R = 230
INNER_R = 90

MUSIC_FILE = "rexlambo-falling-in-love.wav"

# Webcam toggle
webcam_enabled = True   # W1 — ON by default

# ─────────────────────────────────────────────────────────
# SUNSET PASTEL PALETTE
# ─────────────────────────────────────────────────────────

BG_COLOR        = (18, 14, 28)
PANEL_COLOR     = (26, 20, 40)

PEACH           = (247, 198, 163)
LAVENDER        = (215, 180, 243)
SOFT_ORANGE     = (249, 168, 117)
WARM_PINK       = (245, 163, 199)
SOFT_YELLOW     = (248, 222, 160)

WHITE_SOFT      = (235, 232, 245)
WHITE_DIM       = (190, 186, 210)
WHITE_FAINT     = (140, 135, 165)

SEG_BASE        = (34, 26, 52)
SEG_HOVER       = (52, 40, 78)

CURSOR_LEFT     = WARM_PINK
CURSOR_RIGHT    = LAVENDER

# ─────────────────────────────────────────────────────────
# PYGAME INIT
# ─────────────────────────────────────────────────────────

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("Sunset Pastel Reaction Wheel")
clock = pygame.time.Clock()

font_sm   = pygame.font.SysFont("Segoe UI", 18)
font_md   = pygame.font.SysFont("Segoe UI", 24)
font_lg   = pygame.font.SysFont("Segoe UI", 32)
font_xl   = pygame.font.SysFont("Segoe UI", 46)
font_note = pygame.font.SysFont("Segoe UI", 26, bold=True)
font_emoji = pygame.font.SysFont("Segoe UI Emoji", 40)

# ─────────────────────────────────────────────────────────
# GAME STATE
# ─────────────────────────────────────────────────────────

mode    = "rhythm"
playing = False

score = 0
combo = 0
max_combo = 0

feedback_text  = ""
feedback_color = WHITE_SOFT
feedback_timer = 0
FEEDBACK_DUR   = 40

active_idx      = -1
hover_left_idx  = -1
hover_right_idx = -1
target_idx      = -1

flash_slices = [0] * N_SLICES

SLICE_LABELS = ["C", "D", "E", "F", "G", "A", "B", "C2"]
# ─────────────────────────────────────────────────────────
# DIP GESTURE
# ─────────────────────────────────────────────────────────

prev_index_y  = {"left": None, "right": None}
DIP_THRESHOLD = 0.018

def detect_dip(result, side):
    global prev_index_y
    if result is None or not result.hand_landmarks:
        prev_index_y[side] = None
        return False

    for lm_list, handed in zip(result.hand_landmarks, result.handedness):
        label = handed[0].category_name  # "Left" or "Right"

        # After flip: Left = left hand, Right = right hand
        if (label == "Left" and side == "left") or (label == "Right" and side == "right"):
            tip_y = lm_list[8].y
            prev  = prev_index_y[side]
            prev_index_y[side] = tip_y

            if prev is not None and (tip_y - prev) > DIP_THRESHOLD:
                return True

    return False


# ─────────────────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────────────────

BaseOptions        = mp.tasks.BaseOptions
VisionRunningMode  = mp.tasks.vision.RunningMode
HandLandmarker     = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    num_hands=2,
    running_mode=VisionRunningMode.IMAGE,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = HandLandmarker.create_from_options(hand_options)
cam = cv2.VideoCapture(0)


def get_hands():
    ret, frame = cam.read()
    if not ret:
        return None, None, False, False, None

    frame     = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result    = hand_landmarker.detect(mp_image)

    left_pos   = None
    right_pos  = None
    left_pinch = False
    right_pinch= False

    if result.hand_landmarks and result.handedness:
        for lm_list, handed in zip(result.hand_landmarks, result.handedness):
            label     = handed[0].category_name
            idx_tip   = lm_list[8]
            thumb_tip = lm_list[4]

            x = int(idx_tip.x * WIN_W)
            y = int(idx_tip.y * WIN_H)

            dist  = math.hypot(idx_tip.x - thumb_tip.x, idx_tip.y - thumb_tip.y)
            pinch = dist < 0.06

            if label == "Left":
                left_pos   = (x, y)
                left_pinch = pinch
            else:
                right_pos   = (x, y)
                right_pinch = pinch

    return left_pos, right_pos, left_pinch, right_pinch, result


# ─────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────

def draw_text_c(surf, text, font, color, cx, cy):
    img  = font.render(text, True, color)
    rect = img.get_rect(center=(cx, cy))
    surf.blit(img, rect)


def draw_finger_cursor(pos, color, dipping):
    if pos is None:
        return
    x, y = pos
    pygame.draw.circle(screen, color, (x, y), 10, 2)
    if dipping:
        pygame.draw.circle(screen, color, (x, y), 18, 2)


def get_slice_index(px, py):
    dx   = px - CENTER_X
    dy   = py - CENTER_Y
    dist = math.hypot(dx, dy)

    if dist < INNER_R or dist > OUTER_R + 40:
        return -1

    angle = math.atan2(dy, dx)
    norm  = (angle + 2 * math.pi) % (2 * math.pi)
    idx   = int(norm / (2 * math.pi / N_SLICES))
    return idx
# ─────────────────────────────────────────────────────────
# WHEEL DRAWING
# ─────────────────────────────────────────────────────────

def slice_points(i, extra_r=0):
    slice_angle = 2 * math.pi / N_SLICES
    offset      = -math.pi / 2
    start = offset + i * slice_angle
    end   = start + slice_angle

    steps = 32
    pts   = [(CENTER_X, CENTER_Y)]

    for s in range(steps + 1):
        a = start + (end - start) * s / steps
        pts.append((
            CENTER_X + math.cos(a) * (OUTER_R + extra_r),
            CENTER_Y + math.sin(a) * (OUTER_R + extra_r)
        ))

    return pts, (start + end) / 2


def draw_wheel():
    for i in range(N_SLICES):
        is_target = (i == target_idx)
        is_hover  = (i == hover_left_idx or i == hover_right_idx)
        is_active = (i == active_idx)

        pop = 32 if is_target else 0
        pts, mid = slice_points(i, pop)

        # Base fill
        if is_active:
            base_col = (int(WARM_PINK[0]*0.9), int(WARM_PINK[1]*0.9), int(WARM_PINK[2]*0.9))
            alpha    = 200
        elif is_target:
            base_col = SOFT_ORANGE
            alpha    = 190
        elif is_hover:
            base_col = (80, 60, 120)
            alpha    = 180
        else:
            base_col = SEG_BASE
            alpha    = 140

        seg_surf = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        pygame.draw.polygon(seg_surf, (*base_col, alpha), pts)
        screen.blit(seg_surf, (0, 0))

        # Outline
        if is_hover:
            pygame.draw.polygon(screen, WHITE_SOFT, pts, 3)
        elif is_target:
            pygame.draw.polygon(screen, SOFT_ORANGE, pts, 3)
        else:
            pygame.draw.polygon(screen, (60, 50, 90), pts, 1)

        # Flash glow
        if flash_slices[i] > 0:
            glow_surf  = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
            glow_color = WARM_PINK if mode == "rhythm" else SOFT_ORANGE
            galpha     = min(160, 50 + flash_slices[i] * 5)
            pygame.draw.polygon(glow_surf, (*glow_color, galpha), pts)
            screen.blit(glow_surf, (0, 0))
            flash_slices[i] -= 1

        # Label
        label_r = ((OUTER_R + pop) + INNER_R) / 2
        lx = CENTER_X + math.cos(mid) * label_r
        ly = CENTER_Y + math.sin(mid) * label_r

        if is_hover:
            lc    = WHITE_SOFT
            lfont = pygame.font.SysFont("Segoe UI", 32, bold=True)
        elif is_target:
            lc    = SOFT_YELLOW
            lfont = pygame.font.SysFont("Segoe UI", 30, bold=True)
        elif is_active:
            lc    = WARM_PINK
            lfont = font_note
        else:
            lc    = WHITE_DIM
            lfont = font_note

        draw_text_c(screen, SLICE_LABELS[i], lfont, lc, int(lx), int(ly))

    # Rings
    pygame.draw.circle(screen, BG_COLOR,      (CENTER_X, CENTER_Y), INNER_R)
    pygame.draw.circle(screen, (70, 60, 100), (CENTER_X, CENTER_Y), INNER_R, 1)
    pygame.draw.circle(screen, (70, 60, 100), (CENTER_X, CENTER_Y), OUTER_R, 1)

    # Center label
    draw_text_c(screen, "dip to hit", font_sm, WHITE_FAINT, CENTER_X, CENTER_Y)


# ─────────────────────────────────────────────────────────
# FEEDBACK
# ─────────────────────────────────────────────────────────

def set_feedback(text, color):
    global feedback_text, feedback_color, feedback_timer
    feedback_text  = text
    feedback_color = color
    feedback_timer = FEEDBACK_DUR


def draw_feedback():
    if feedback_timer > 0 and feedback_text:
        alpha = min(255, feedback_timer * 6)
        surf  = font_xl.render(feedback_text, True, feedback_color)
        surf.set_alpha(alpha)
        screen.blit(surf, surf.get_rect(center=(WIN_W // 2, 90)))
# ─────────────────────────────────────────────────────────
# RHYTHM MODE
# ─────────────────────────────────────────────────────────

beat_interval  = 900
last_beat_time = 0

def fire_rhythm_beat(now):
    global target_idx, last_beat_time
    target_idx = random.randint(0, N_SLICES - 1)
    flash_slices[target_idx] = 20
    last_beat_time = now

def update_rhythm_mode(now):
    if not playing:
        return
    if now - last_beat_time >= beat_interval:
        fire_rhythm_beat(now)


# ─────────────────────────────────────────────────────────
# REACTION MODE
# ─────────────────────────────────────────────────────────

reaction_target      = -1
reaction_start_time  = 0
reaction_window      = 700
reaction_ready       = True

reaction_interval     = 1500
reaction_min_interval = 900
reaction_max_interval = 2000

reaction_times   = []
reaction_hits    = 0
reaction_misses  = 0
best_reaction    = None
avg_reaction     = None

reaction_feedback       = ""
reaction_feedback_timer = 0
REACTION_FEEDBACK_DUR   = 40


def start_reaction_target(now):
    global reaction_target, reaction_start_time, reaction_ready
    reaction_target      = random.randint(0, N_SLICES - 1)
    reaction_start_time  = now
    reaction_ready       = False


def update_reaction_mode(now, hover_idx, dipping):
    global reaction_ready, reaction_target
    global reaction_times, reaction_hits, reaction_misses
    global best_reaction, avg_reaction
    global reaction_interval, reaction_feedback, reaction_feedback_timer

    # Spawn new target
    if reaction_ready:
        start_reaction_target(now)

    # Hit detection
    if dipping and hover_idx == reaction_target:
        rt = now - reaction_start_time

        if rt <= reaction_window:
            reaction_hits += 1
            reaction_times.append(rt)

            best_reaction = min(reaction_times)
            avg_reaction  = sum(reaction_times) / len(reaction_times)

            if rt < 300:   reaction_feedback = "Lightning fast!"
            elif rt < 400: reaction_feedback = "Great!"
            elif rt < 500: reaction_feedback = "Nice!"
            else:          reaction_feedback = "Good!"

            reaction_feedback_timer = REACTION_FEEDBACK_DUR

            # Difficulty adjust
            if rt < 350:
                reaction_interval = max(reaction_min_interval, reaction_interval - 80)
            elif rt > 550:
                reaction_interval = min(reaction_max_interval, reaction_interval + 80)

        else:
            reaction_misses += 1
            reaction_feedback = "Late!"
            reaction_feedback_timer = REACTION_FEEDBACK_DUR

        reaction_ready  = True
        reaction_target = -1
        return

    # Missed window
    if now - reaction_start_time > reaction_window and not reaction_ready:
        reaction_misses += 1
        reaction_feedback = "Miss!"
        reaction_feedback_timer = REACTION_FEEDBACK_DUR
        reaction_ready  = True
        reaction_target = -1


# ─────────────────────────────────────────────────────────
# HIT HANDLER
# ─────────────────────────────────────────────────────────

def handle_hit(idx):
    global active_idx, score, combo, target_idx, max_combo

    if mode == "rhythm":
        if idx == target_idx:
            score += 10 + combo
            combo += 1

            if combo > max_combo:
                max_combo = combo

            set_feedback("Perfect!", WARM_PINK)
            target_idx = -1
        else:
            combo = 0
            set_feedback("Miss!", SOFT_ORANGE)

    else:
        set_feedback(SLICE_LABELS[idx], LAVENDER)

    active_idx = idx
    flash_slices[idx] = 15
# ─────────────────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────────────────

def draw_hud():
    pygame.draw.rect(screen, PANEL_COLOR, (0, 0, WIN_W, 60))

    draw_text_c(screen, f"Mode: {mode}",   font_md, WHITE_SOFT, 100,         30)
    draw_text_c(screen, f"Score: {score}", font_md, PEACH,      WIN_W // 2,  30)
    draw_text_c(screen, f"Combo: {combo}", font_md, WARM_PINK,  WIN_W - 120, 30)

    hints = "SPACE: play/pause   M: toggle mode   S: summary   W: webcam   ESC: quit"
    draw_text_c(screen, hints, font_sm, WHITE_FAINT, WIN_W // 2, WIN_H - 14)


# ─────────────────────────────────────────────────────────
# REACTION TIMER + FEEDBACK
# ─────────────────────────────────────────────────────────

def draw_reaction_timer(now):
    if reaction_target < 0:
        return

    elapsed   = now - reaction_start_time
    remaining = max(0, reaction_window - elapsed)
    pct       = remaining / reaction_window

    bar_w = int(300 * pct)
    x, y  = WIN_W // 2 - 150, 75

    pygame.draw.rect(screen, (40, 30, 60), (x, y, 300, 12), border_radius=6)
    pygame.draw.rect(screen, SOFT_ORANGE,  (x, y, bar_w, 12), border_radius=6)


def draw_reaction_feedback():
    if reaction_feedback_timer > 0 and reaction_feedback:
        alpha = min(255, reaction_feedback_timer * 6)
        surf  = font_xl.render(reaction_feedback, True, SOFT_YELLOW)
        surf.set_alpha(alpha)
        screen.blit(surf, surf.get_rect(center=(WIN_W // 2, 140)))


# ─────────────────────────────────────────────────────────
# GAME SUMMARY (Option A — Rhythm + Reaction)
# ─────────────────────────────────────────────────────────

def get_performance_message(metric):
    if metric == 0:
        msg = "💀💔 Boo... Try again!"
        color = (255, 80, 80)
    elif metric < 30:
        msg = "💔 Keep going, you’ll get better!"
        color = (255, 120, 120)
    elif metric < 50:
        msg = "✨ You got this! Keep practicing!"
        color = (255, 180, 120)
    elif metric < 70:
        msg = "🌟 Nice progress! You're improving!"
        color = (255, 210, 150)
    elif metric < 80:
        msg = "🔥 Great job! You're getting sharp!"
        color = (255, 230, 180)
    elif metric < 90:
        msg = "💙💚 Amazing! You're fast!"
        color = (180, 255, 220)
    elif metric < 100:
        msg = "💚💙 Incredible! Almost perfect!"
        color = (150, 255, 200)
    else:
        msg = "💚💙💚 PERFECT! 100% Accuracy! 💙💚💙"
        color = (120, 255, 180)

    return msg, color


def draw_rhythm_summary():
    draw_text_c(screen, "Rhythm Mode", font_lg, SOFT_ORANGE, WIN_W//2, 150)
    draw_text_c(screen, f"Score: {score}", font_md, PEACH, WIN_W//2, 200)
    draw_text_c(screen, f"Max Combo: {max_combo}", font_md, WARM_PINK, WIN_W//2, 240)

    rhythm_metric = min(100, score / 2)
    rhythm_msg, rhythm_color = get_performance_message(rhythm_metric)
    draw_text_c(screen, rhythm_msg, font_emoji, rhythm_color, WIN_W//2, 300)


def draw_summary():
    screen.fill(BG_COLOR)
    draw_text_c(screen, "GAME SUMMARY", font_xl, SOFT_YELLOW, WIN_W//2, 80)

    draw_rhythm_summary()

    draw_text_c(screen, "Press any key to continue", font_md, WHITE_DIM, WIN_W//2, WIN_H - 80)
    pygame.display.flip()
# ─────────────────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────────────────

def main():
    global mode, playing
    global active_idx, hover_left_idx, hover_right_idx
    global target_idx, feedback_timer
    global reaction_feedback_timer, reaction_ready, reaction_target
    global last_beat_time, webcam_enabled

    summary_mode = False

    while True:
        clock.tick(FPS)
        now = pygame.time.get_ticks()

        # ─────────────────────────────────────────
        # SUMMARY SCREEN
        # ─────────────────────────────────────────
        if summary_mode:
            draw_summary()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN:
                    summary_mode = False

            continue

        # ─────────────────────────────────────────
        # HAND TRACKING
        # ─────────────────────────────────────────
        if webcam_enabled:
            left_pos, right_pos, left_pinch, right_pinch, hand_data_raw = get_hands()
        else:
            left_pos = right_pos = None
            left_pinch = right_pinch = False
            hand_data_raw = None

        hover_left_idx  = get_slice_index(*left_pos)  if left_pos  else -1
        hover_right_idx = get_slice_index(*right_pos) if right_pos else -1

        left_dip  = detect_dip(hand_data_raw, "left")  if webcam_enabled else False
        right_dip = detect_dip(hand_data_raw, "right") if webcam_enabled else False

        if left_dip and hover_left_idx >= 0:
            handle_hit(hover_left_idx)

        if right_dip and hover_right_idx >= 0:
            handle_hit(hover_right_idx)

        # ─────────────────────────────────────────
        # MODE UPDATE
        # ─────────────────────────────────────────
        if mode == "rhythm":
            update_rhythm_mode(now)
        else:
            hover_idx = hover_left_idx if hover_left_idx >= 0 else hover_right_idx
            dipping   = left_dip or right_dip
            update_reaction_mode(now, hover_idx, dipping)

        # ─────────────────────────────────────────
        # TIMERS
        # ─────────────────────────────────────────
        if feedback_timer > 0:
            feedback_timer -= 1

        if reaction_feedback_timer > 0:
            reaction_feedback_timer -= 1

        # ─────────────────────────────────────────
        # DRAW BACKGROUND (WEBCAM OR SOLID)
        # ─────────────────────────────────────────
        if webcam_enabled:
            ret, frame = cam.read()
            if ret:
                frame = cv2.flip(frame, 1)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                surf = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                surf = pygame.transform.scale(surf, (WIN_W, WIN_H))
                screen.blit(surf, (0, 0))
            else:
                screen.fill(BG_COLOR)
        else:
            screen.fill(BG_COLOR)

        # ─────────────────────────────────────────
        # DRAW GAME ELEMENTS
        # ─────────────────────────────────────────
        draw_wheel()
        draw_hud()
        draw_feedback()

        if left_pos:
            draw_finger_cursor(left_pos, CURSOR_LEFT, left_dip)

        if right_pos:
            draw_finger_cursor(right_pos, CURSOR_RIGHT, right_dip)

        if mode == "reaction":
            draw_reaction_timer(now)
            draw_reaction_feedback()

        pygame.display.flip()

        # ─────────────────────────────────────────
        # EVENTS
        # ─────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()

                # Play / Pause
                if event.key == pygame.K_SPACE:
                    if not playing:
                        playing = True
                        pygame.mixer.music.load(MUSIC_FILE)
                        pygame.mixer.music.play()
                        last_beat_time = now
                    else:
                        playing = False
                        pygame.mixer.music.pause()

                # Toggle mode
                if event.key == pygame.K_m:
                    mode = "reaction" if mode == "rhythm" else "rhythm"
                    active_idx      = -1
                    target_idx      = -1
                    reaction_ready  = True
                    reaction_target = -1

                # Summary
                if event.key == pygame.K_s:
                    summary_mode = True

                # Webcam toggle
                if event.key == pygame.K_w:
                    webcam_enabled = not webcam_enabled
# ─────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
