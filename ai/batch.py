#!/usr/bin/env python3
"""
Viviti AI Batch — blur, duplicate, and scene detection.
Usage: python3 batch.py <photo_dir> <db_path>
Requires: pip3 install opencv-python imagehash Pillow tflite-runtime
"""
import sys, os, sqlite3, json
from pathlib import Path
import cv2
import numpy as np

try:
    import imagehash
    from PIL import Image
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

try:
    import scene as scene_mod
    HAS_SCENE = True
except ImportError:
    HAS_SCENE = False

IMAGE_EXT      = {'.jpg', '.jpeg', '.png', '.heic', '.cr2', '.arw', '.nef', '.dng'}
BLUR_THRESHOLD = 100
DUPE_THRESHOLD = 8


def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS photo_ai (
            photo_path   TEXT PRIMARY KEY,
            processed_at TEXT DEFAULT (datetime('now')),
            blur_score   REAL,
            is_blurry    INTEGER DEFAULT 0,
            phash        TEXT,
            is_duplicate INTEGER DEFAULT 0,
            duplicate_of TEXT,
            scene_tags   TEXT,
            objects      TEXT,
            ai_score     REAL
        )
    ''')
    # Add columns if upgrading from older schema
    for col, typedef in [('scene_tags', 'TEXT'), ('objects', 'TEXT')]:
        try:
            conn.execute(f'ALTER TABLE photo_ai ADD COLUMN {col} {typedef}')
        except Exception:
            pass
    conn.commit()
    return conn


def blur_score(abs_path):
    img = cv2.imread(abs_path)
    if img is None:
        return None
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def compute_phash(abs_path):
    if not HAS_IMAGEHASH:
        return None
    try:
        return str(imagehash.phash(Image.open(abs_path)))
    except Exception:
        return None


def run_batch(photo_dir, db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = init_db(db_path)

    all_photos = []
    for root, _, files in os.walk(photo_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXT:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, photo_dir).replace('\\', '/')
                all_photos.append((rel_path, abs_path))

    already_done = {r[0] for r in conn.execute('SELECT photo_path FROM photo_ai')}
    to_process   = [(r, a) for r, a in all_photos if r not in already_done]

    if not to_process:
        print(json.dumps({'processed': 0, 'blurry': 0, 'duplicates': 0, 'tagged': 0}))
        return

    existing_hashes = {}
    if HAS_IMAGEHASH:
        for row in conn.execute('SELECT photo_path, phash FROM photo_ai WHERE phash IS NOT NULL'):
            existing_hashes[row[0]] = row[1]

    # Load scene detector once for the whole batch
    detector = scene_mod.SceneDetector() if HAS_SCENE else None

    counts = {'processed': 0, 'blurry': 0, 'duplicates': 0, 'tagged': 0}

    for rel_path, abs_path in to_process:
        score = blur_score(abs_path)
        if score is None:
            continue

        is_blurry    = 1 if score < BLUR_THRESHOLD else 0
        ph           = compute_phash(abs_path)
        is_duplicate = 0
        duplicate_of = None

        if ph and HAS_IMAGEHASH:
            ph_hash = imagehash.hex_to_hash(ph)
            for ep, eph in existing_hashes.items():
                try:
                    if ph_hash - imagehash.hex_to_hash(eph) < DUPE_THRESHOLD:
                        is_duplicate = 1
                        duplicate_of = ep
                        break
                except Exception:
                    pass
            if not is_duplicate:
                existing_hashes[rel_path] = ph

        scene_tags = objects = None
        if detector:
            try:
                result     = detector.analyse(abs_path)
                scene_tags = json.dumps(result.get('scene_tags', []))
                objects    = json.dumps(result.get('objects', []))
                if result.get('scene_tags'):
                    counts['tagged'] += 1
            except Exception as e:
                print(f'[scene] {rel_path}: {e}', file=sys.stderr)

        conn.execute('''
            INSERT OR REPLACE INTO photo_ai
            (photo_path, blur_score, is_blurry, phash, is_duplicate, duplicate_of, scene_tags, objects)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (rel_path, score, is_blurry, ph, is_duplicate, duplicate_of, scene_tags, objects))

        counts['processed'] += 1
        if is_blurry:    counts['blurry']     += 1
        if is_duplicate: counts['duplicates'] += 1

        if counts['processed'] % 50 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    print(json.dumps(counts))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: batch.py <photo_dir> <db_path>'}))
        sys.exit(1)
    run_batch(sys.argv[1], sys.argv[2])
