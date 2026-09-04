"""Codec availability test for VideoWriter on this OpenCV install.

Run: python test_codec.py
Reports which fourcc codes produce a decodable file at 2560x1440 @ 1 fps.
"""
import cv2
import numpy as np
import os
import time

W, H, FPS = 2560, 1440, 1
N_FRAMES = 5


def test_codec(fourcc_str, ext):
    out_path = f"_codec_test_{fourcc_str}{ext}"
    if os.path.exists(out_path):
        os.remove(out_path)

    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    t0 = time.time()
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))
    if not writer.isOpened():
        return False, "VideoWriter would not open"
    try:
        for _ in range(N_FRAMES):
            # Random noise prevents trivial compression — gives a realistic file size.
            frame = np.random.randint(50, 200, (H, W, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    elapsed = time.time() - t0

    if not os.path.exists(out_path):
        return False, "no output file produced"
    size = os.path.getsize(out_path)
    if size < 2048:
        os.remove(out_path)
        return False, f"file too small ({size} B) — codec silently failed"

    # Decode the file back to confirm it's valid.
    cap = cv2.VideoCapture(out_path)
    n_read = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        n_read += 1
    cap.release()
    os.remove(out_path)
    return True, f"{size // 1024} KB, decoded {n_read}/{N_FRAMES} frames, {elapsed * 1000:.0f} ms"


CANDIDATES = [
    ("avc1", ".mp4"),  # H.264 (preferred — small, high quality)
    ("H264", ".mp4"),  # same codec, alternate fourcc
    ("X264", ".mp4"),  # libx264
    ("mp4v", ".mp4"),  # MPEG-4 Part 2 (current renderer uses this — bulky)
    ("MJPG", ".avi"),  # Motion JPEG (every frame is a JPEG — crash-safe)
    ("XVID", ".avi"),  # XVID
]

print(f"Codec test @ {W}x{H}, {N_FRAMES} frames, {FPS} fps")
print("-" * 78)
for fourcc, ext in CANDIDATES:
    print(f"  {fourcc:6s} ({ext}): ", end="", flush=True)
    try:
        ok, info = test_codec(fourcc, ext)
        print(f"{'OK  ' if ok else 'FAIL'}  {info}")
    except Exception as e:
        print(f"ERROR  {e!r}")
