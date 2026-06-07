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

    # Time-based seek (POS_MSEC) is far more reliable across codecs than frame
    # index. Try ~1s in (skip black intros), fall back to the very start.
    frame = None
    for ms in (1000, 0):
        cap.set(cv2.CAP_PROP_POS_MSEC, ms)
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
            break
    if frame is None:
        ok, f = cap.read()          # last resort: sequential first frame
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        return False

    try:
        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return False
        scale = size / min(h, w)
        rw, rh = max(size, int(w * scale)), max(size, int(h * scale))
        resized = cv2.resize(frame, (rw, rh))
        x = (rw - size) // 2
        y = (rh - size) // 2
        crop = resized[y:y + size, x:x + size]
        return bool(cv2.imwrite(out_path, crop, [cv2.IMWRITE_JPEG_QUALITY, 82]))
    except Exception:
        return False


if __name__ == '__main__':
    if len(sys.argv) < 4:
        sys.exit(1)
    ok = make_thumb(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    sys.exit(0 if ok else 1)
