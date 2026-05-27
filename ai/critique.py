#!/usr/bin/env python3
"""
Viviti Photo Critique — on-demand quality analysis.
Usage: python3 critique.py <absolute_image_path>
Output: JSON to stdout
Requires: pip install opencv-python
"""
import sys, json, math
import cv2
import numpy as np


def analyse(path):
    img = cv2.imread(path)
    if img is None:
        return {'error': 'Cannot read image'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    issues = []

    # ── Blur / focus ─────────────────────────────────────────────────────────
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap < 40:
        issues.append({'type': 'blur', 'sev': 'high',
                       'msg': 'Subject is out of focus — try manual focus or a faster shutter'})
    elif lap < 100:
        issues.append({'type': 'blur', 'sev': 'medium',
                       'msg': 'Slight blur detected — try a faster shutter speed'})

    # ── Exposure — underexposed ───────────────────────────────────────────────
    mean_bright = float(gray.mean())
    if mean_bright < 45:
        issues.append({'type': 'underexposed', 'sev': 'high',
                       'msg': 'Image is very underexposed — raise ISO or use a longer exposure'})
    elif mean_bright < 75:
        issues.append({'type': 'underexposed', 'sev': 'medium',
                       'msg': 'Image appears dark — try +1 stop exposure compensation'})

    # ── Exposure — overexposed ────────────────────────────────────────────────
    clipped = float((gray > 248).mean() * 100)
    if clipped > 5:
        issues.append({'type': 'overexposed', 'sev': 'high',
                       'msg': f'Highlights blown out ({clipped:.0f}% clipped) — reduce exposure'})
    elif clipped > 1:
        issues.append({'type': 'overexposed', 'sev': 'medium',
                       'msg': 'Some highlight clipping — reduce exposure slightly'})

    # ── Noise ─────────────────────────────────────────────────────────────────
    noise = _noise_sigma(gray)
    if noise > 20:
        issues.append({'type': 'noise', 'sev': 'high',
                       'msg': 'Heavy noise / grain — shoot at a lower ISO or in better light'})
    elif noise > 10:
        issues.append({'type': 'noise', 'sev': 'medium',
                       'msg': 'Moderate noise visible — try lowering ISO'})

    # ── Horizon tilt (landscape photos only) ─────────────────────────────────
    if w > h * 1.1:
        tilt = _horizon_tilt(gray)
        if tilt is not None and abs(tilt) > 1.5:
            side = 'right' if tilt > 0 else 'left'
            issues.append({'type': 'tilt', 'sev': 'medium',
                           'msg': f'Horizon tilted {abs(tilt):.1f}° to the {side} — straighten the camera'})

    # ── White balance ─────────────────────────────────────────────────────────
    b = float(img[:, :, 0].mean())
    g = float(img[:, :, 1].mean())
    r = float(img[:, :, 2].mean())
    if r > g * 1.25 and r > b * 1.3:
        issues.append({'type': 'whitebalance', 'sev': 'low',
                       'msg': 'Warm colour cast (orange / yellow) — adjust white balance'})
    elif b > g * 1.25 and b > r * 1.3:
        issues.append({'type': 'whitebalance', 'sev': 'low',
                       'msg': 'Cool colour cast (blue) — adjust white balance'})

    # ── Composition ───────────────────────────────────────────────────────────
    edges = cv2.Canny(gray, 50, 150)
    tot = float(edges.sum()) or 1.0
    mid_h = edges[h // 3: 2 * h // 3, :].sum() / tot
    mid_v = edges[:, w // 3: 2 * w // 3].sum() / tot
    if mid_h > 0.55 and mid_v > 0.55:
        issues.append({'type': 'composition', 'sev': 'low',
                       'msg': 'Subject appears centred — try placing it on a rule-of-thirds intersection'})

    # ── Score ─────────────────────────────────────────────────────────────────
    deductions = {'high': 25, 'medium': 12, 'low': 4}
    score = 100 - sum(deductions.get(i['sev'], 0) for i in issues)
    score = max(0, min(100, score))

    if not issues:
        issues.append({'type': 'ok', 'sev': 'none',
                       'msg': 'Great shot — no major issues detected'})

    return {
        'score': score,
        'blur_score': round(lap, 1),
        'brightness': round(mean_bright, 1),
        'noise': round(float(noise), 1),
        'issues': issues,
    }


def _noise_sigma(gray):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    h, w = gray.shape
    filtered = cv2.filter2D(gray.astype(np.float64), -1, k)
    return float(np.sum(np.abs(filtered)) * math.sqrt(0.5 * math.pi)
                 / (6 * max(w - 2, 1) * max(h - 2, 1)))


def _horizon_tilt(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=80)
    if lines is None:
        return None
    angles = []
    for line in lines:
        deg = float(np.degrees(line[0][1]) - 90)
        if abs(deg) < 20:
            angles.append(deg)
    return float(np.median(angles)) if angles else None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: critique.py <image_path>'}))
        sys.exit(1)
    print(json.dumps(analyse(sys.argv[1])))
