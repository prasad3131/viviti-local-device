#!/usr/bin/env python3
"""
Viviti Photo Critique — structured 7-point analysis.
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

    # ── Raw measurements ──────────────────────────────────────────────────────
    lap         = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_bright = float(gray.mean())
    clipped     = float((gray > 248).mean() * 100)
    noise       = _noise_sigma(gray)
    b_mean      = float(img[:, :, 0].mean())
    g_mean      = float(img[:, :, 1].mean())
    r_mean      = float(img[:, :, 2].mean())

    # ── Orientation & aspect ratio ────────────────────────────────────────────
    orientation = 'landscape' if w > h else ('portrait' if h > w else 'square')
    gcd = math.gcd(w, h)
    aspect_ratio = f'{w // gcd}:{h // gcd}'

    # ── Interpretation — mood ─────────────────────────────────────────────────
    warm_bias = r_mean - b_mean
    if warm_bias > 20:
        mood = 'warm'
        mood_desc = ('This photo has warm tones — oranges, reds, and yellows dominate. '
                     'It may evoke feelings of energy, comfort, or nostalgia.')
    elif warm_bias < -20:
        mood = 'cool'
        mood_desc = ('This photo has cool tones — blues and greens dominate. '
                     'It may evoke feelings of calm, distance, or melancholy.')
    else:
        mood = 'neutral'
        mood_desc = ('This photo has a neutral, balanced colour tone. '
                     'It reads as natural and unbiased.')

    if mean_bright > 170:
        mood_desc += ' The high-key lighting feels open and airy.'
    elif mean_bright < 80:
        mood_desc += ' The low-key lighting adds drama and mood.'

    # ── Interpretation — composition feel ─────────────────────────────────────
    edges   = cv2.Canny(gray, 50, 150)
    tot     = float(edges.sum()) or 1.0
    mid_h   = edges[h // 3: 2 * h // 3, :].sum() / tot
    mid_v   = edges[:, w // 3: 2 * w // 3].sum() / tot
    centred = mid_h > 0.55 and mid_v > 0.55

    if centred:
        composition_feel = 'The subject appears centred, giving a formal, symmetrical feel.'
    else:
        composition_feel = 'The subject placement creates an asymmetric, dynamic feel.'

    # ── Technical issues ──────────────────────────────────────────────────────
    technical = []

    if lap < 40:
        technical.append({'sev': 'high',
                          'msg': 'Subject is out of focus — try manual focus or a faster shutter speed'})
    elif lap < 100:
        technical.append({'sev': 'medium',
                          'msg': 'Slight blur detected — try a faster shutter speed to freeze motion'})

    if mean_bright < 45:
        technical.append({'sev': 'high',
                          'msg': 'Very underexposed — raise ISO or use a longer exposure'})
    elif mean_bright < 75:
        technical.append({'sev': 'medium',
                          'msg': 'Image appears dark — try +1 stop exposure compensation'})

    if clipped > 5:
        technical.append({'sev': 'high',
                          'msg': f'Highlights blown out ({clipped:.0f}% clipped) — reduce exposure by 1–2 stops'})
    elif clipped > 1:
        technical.append({'sev': 'medium',
                          'msg': 'Some highlight clipping — reduce exposure slightly'})

    if noise > 20:
        technical.append({'sev': 'high',
                          'msg': 'Heavy noise / grain — shoot at a lower ISO or in better light'})
    elif noise > 10:
        technical.append({'sev': 'medium',
                          'msg': 'Moderate noise visible — try lowering ISO'})

    if w > h * 1.1:
        tilt = _horizon_tilt(gray)
        if tilt is not None and abs(tilt) > 1.5:
            side = 'right' if tilt > 0 else 'left'
            technical.append({'sev': 'medium',
                              'msg': f'Horizon tilted {abs(tilt):.1f}° to the {side} — straighten in editing'})

    if r_mean > g_mean * 1.25 and r_mean > b_mean * 1.3:
        technical.append({'sev': 'low',
                          'msg': 'Warm colour cast (orange/yellow) — try a cooler white balance setting'})
    elif b_mean > g_mean * 1.25 and b_mean > r_mean * 1.3:
        technical.append({'sev': 'low',
                          'msg': 'Cool colour cast (blue) — try a warmer white balance setting'})

    # ── Artistic issues ───────────────────────────────────────────────────────
    artistic = []

    if centred:
        artistic.append({'sev': 'low',
                         'msg': 'Subject is centred — try placing it on a rule-of-thirds intersection for more energy'})

    if w > h * 2.2:
        artistic.append({'sev': 'low',
                         'msg': 'Very wide aspect ratio — consider whether a panoramic crop serves the subject'})

    # Colour variety
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat  = float(hsv[:, :, 1].mean())
    if sat < 30:
        artistic.append({'sev': 'low',
                         'msg': 'Very low colour saturation — consider a black & white conversion for more impact'})

    # ── Good points ───────────────────────────────────────────────────────────
    good_points = []

    if lap >= 200:
        good_points.append('Excellent sharpness — the subject is crisp and well-focused')
    elif lap >= 100:
        good_points.append('Acceptable sharpness — focus is generally good')

    if 75 <= mean_bright <= 185 and clipped < 1:
        good_points.append('Well-balanced exposure — good detail in highlights and shadows')

    if noise < 5:
        good_points.append('Very clean image — low noise, likely shot at a low ISO')
    elif noise < 10:
        good_points.append('Good noise levels — image is clean')

    if not centred:
        good_points.append('Effective composition — the off-centre placement adds visual interest')

    if sat >= 60:
        good_points.append('Strong, vibrant colours that draw the eye')

    if not good_points:
        good_points.append('The photo captures its subject clearly')

    # ── Improvements ─────────────────────────────────────────────────────────
    improvements = [i for i in technical + artistic if i['sev'] in ('high', 'medium')]

    # ── Score ─────────────────────────────────────────────────────────────────
    deductions = {'high': 25, 'medium': 12, 'low': 4}
    all_issues = technical + artistic
    score = 100 - sum(deductions.get(i['sev'], 0) for i in all_issues)
    score = max(0, min(100, score))

    # ── Overall summary ───────────────────────────────────────────────────────
    if score >= 85:
        overall = 'Outstanding shot — excellent technical quality across the board.'
    elif score >= 70:
        overall = 'Good photo — solid technique with a few areas worth refining.'
    elif score >= 55:
        overall = 'Decent shot with some technical issues that could be addressed in editing or future shots.'
    elif score >= 40:
        overall = 'The photo has potential but several technical issues are holding it back.'
    else:
        overall = 'Significant issues detected — use this feedback to strengthen your next shot.'

    return {
        'score':        score,
        'blur_score':   round(lap, 1),
        'brightness':   round(mean_bright, 1),
        'noise':        round(float(noise), 1),
        'orientation':  orientation,
        'aspect_ratio': aspect_ratio,
        'mood':         mood,
        'mood_desc':    mood_desc,
        'composition_feel': composition_feel,
        'technical':    technical,
        'artistic':     artistic,
        'good_points':  good_points,
        'improvements': improvements,
        'overall':      overall,
        'issues':       all_issues,
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
