"""
magic_shield.py
Real-time "Doctor Strange" shield effect using MediaPipe + OpenCV + NumPy.
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os

mp_hands = mp.solutions.hands

# =============================
# Utility Functions
# =============================

def is_palm_open(landmarks, width, height):
    """Simple palm-open detector: checks if at least 3 fingers are extended."""
    tips = [4, 8, 12, 16, 20]  # thumb + fingertips
    cx = int(landmarks.landmark[0].x * width)
    cy = int(landmarks.landmark[0].y * height)

    count_extended = 0
    for t in tips[1:]:  # skip thumb
        fx = int(landmarks.landmark[t].x * width)
        fy = int(landmarks.landmark[t].y * height)
        if fy < cy:  # fingertip above palm center
            count_extended += 1

    return count_extended >= 3


def alpha_blend(img, overlay, x=0, y=0):
    """Blend overlay (with alpha channel) onto img at (x,y), safely clipped."""
    h, w = overlay.shape[:2]

    # Clip overlay to fit inside image
    if x < 0:
        overlay = overlay[:, -x:]
        w = overlay.shape[1]
        x = 0
    if y < 0:
        overlay = overlay[-y:, :]
        h = overlay.shape[0]
        y = 0
    if x + w > img.shape[1]:
        w = img.shape[1] - x
        overlay = overlay[:, :w]
    if y + h > img.shape[0]:
        h = img.shape[0] - y
        overlay = overlay[:h, :]

    if w <= 0 or h <= 0:
        return img

    if overlay.shape[2] == 3:  # no alpha channel
        img[y:y+h, x:x+w] = overlay
        return img

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        img[y:y+h, x:x+w, c] = (
            alpha * overlay[:h, :w, c] +
            (1 - alpha) * img[y:y+h, x:x+w, c]
        )
    return img

# =============================
# Main
# =============================

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Load Doctor Strange shield PNG (RGBA)
    glyph = None
    if os.path.exists("magic_circle.png"):
        glyph = cv2.imread("magic_circle.png", cv2.IMREAD_UNCHANGED)
        if glyph is None or glyph.shape[2] != 4:
            print("Warning: magic_circle.png not valid RGBA, using procedural rings instead")
            glyph = None
    else:
        print("magic_circle.png not found → using procedural shield")

    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.5) as hands:
        prev_time = time.time()
        ring_phase = 0.0

        # Track expansion animation per hand
        hand_timers = {}  # key = hand index, value = time palm opened

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            now = time.time()
            dt = max(0.0, now - prev_time)
            prev_time = now
            ring_phase += dt * 1.8

            if result.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    palm_open = is_palm_open(hand_landmarks, width, height)
                    if palm_open and i not in hand_timers:
                        hand_timers[i] = now  # start animation
                    elif not palm_open and i in hand_timers:
                        del hand_timers[i]

                    if not palm_open:
                        continue  # skip if palm not open

                    # Palm center
                    wrist = hand_landmarks.landmark[0]
                    idx_mcp = hand_landmarks.landmark[5]
                    pinky_mcp = hand_landmarks.landmark[17]
                    cx = int((wrist.x + idx_mcp.x + pinky_mcp.x) / 3 * width)
                    cy = int((wrist.y + idx_mcp.y + pinky_mcp.y) / 3 * height)

                    # Estimate radius from bounding box
                    pts = [(int(lm.x * width), int(lm.y * height)) for lm in hand_landmarks.landmark]
                    pts_np = np.array(pts, dtype=np.int32)
                    x_min, y_min = np.min(pts_np, axis=0)
                    x_max, y_max = np.max(pts_np, axis=0)
                    base_radius = int(0.8 * 0.5 * math.hypot(x_max - x_min, y_max - y_min))
                    base_radius = max(40, min(base_radius, min(width, height)//2 - 10))

                    # Expansion animation
                    anim_progress = min(1.0, (now - hand_timers.get(i, now)) / 0.3)
                    radius = int(base_radius * anim_progress)

                    if glyph is not None:
                        # Resize glyph to hand size
                        scale = (2*radius) / glyph.shape[1]
                        new_size = (int(glyph.shape[1]*scale), int(glyph.shape[0]*scale))
                        if new_size[0] <= 0 or new_size[1] <= 0:
                            continue
                        shield_resized = cv2.resize(glyph, new_size, interpolation=cv2.INTER_AREA)

                        # Rotation
                        angle = (ring_phase * 50) % 360
                        M = cv2.getRotationMatrix2D((new_size[0]//2, new_size[1]//2), angle, 1.0)
                        shield_rotated = cv2.warpAffine(
                            shield_resized, M, new_size[:2],
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0,0,0,0)
                        )

                        # Overlay at palm
                        x0 = cx - new_size[0]//2
                        y0 = cy - new_size[1]//2
                        frame = alpha_blend(frame, shield_rotated, x0, y0)
                    else:
                        # Fallback procedural shield
                        overlay = np.zeros((height, width, 4), dtype=np.uint8)
                        cv2.circle(overlay, (cx, cy), radius, (120,200,255,180), 3, lineType=cv2.LINE_AA)
                        cv2.circle(overlay, (cx, cy), int(radius*0.7), (120,200,255,120), 2, lineType=cv2.LINE_AA)
                        frame = alpha_blend(frame, overlay, 0, 0)

            # HUD
            cv2.putText(frame, "Doctor Strange Shield (press 'q' to quit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow("Magic Shield", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
