#!/usr/bin/env python3
"""
Extract a poster frame from a video and write it as a square JPEG thumbnail.
Usage: python3 video_thumb.py <video_path> <out_path> <size>
"""
import sys
import cv2


def make_thumb(video_path, out_path, size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    # Seek ~1s in (or 10% through) to skip black intro frames.
    target = 0
    if total > 0:
        target = min(int(fps * 1), int(total * 0.1))
    if target > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False

    h, w = frame.shape[:2]
    scale = size / min(h, w)
    rw, rh = max(size, int(w * scale)), max(size, int(h * scale))
    resized = cv2.resize(frame, (rw, rh))
    x = (rw - size) // 2
    y = (rh - size) // 2
    crop = resized[y:y + size, x:x + size]
    cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return True


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(1)
    ok = make_thumb(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    sys.exit(0 if ok else 1)
