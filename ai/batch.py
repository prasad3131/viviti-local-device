#!/usr/bin/env python3
"""
Viviti AI Batch — blur detection + duplicate detection.
Usage: python3 batch.py <photo_dir> <db_path>
Requires: pip install opencv-python imagehash Pillow
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

IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.heic', '.cr2', '.arw', '.nef', '.dng'}
BLUR_THRESHOLD = 100   # Laplacian variance — below this = blurry
DUPE_THRESHOLD = 8     # pHash Hamming distance — below this = duplicate


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
            ai_score     REAL
        )
    ''')
    conn.commit()
    return conn


def blur_score(abs_path):
    img = cv2.imread(abs_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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

    # Collect all photos
    all_photos = []
    for root, _, files in os.walk(photo_dir):
        for f in files:
            if Path(f).suffix.lower() in IMAGE_EXT:
                abs_path = os.path.join(root, f)
                rel_path = os.path.relpath(abs_path, photo_dir)
                all_photos.append((rel_path, abs_path))

    already_done = {r[0] for r in conn.execute('SELECT photo_path FROM photo_ai')}
    to_process = [(r, a) for r, a in all_photos if r not in already_done]

    if not to_process:
        print(json.dumps({'processed': 0, 'blurry': 0, 'duplicates': 0}))
        return

    # Load existing hashes for duplicate detection
    existing_hashes = {}
    if HAS_IMAGEHASH:
        for row in conn.execute('SELECT photo_path, phash FROM photo_ai WHERE phash IS NOT NULL'):
            existing_hashes[row[0]] = row[1]

    counts = {'processed': 0, 'blurry': 0, 'duplicates': 0}

    for rel_path, abs_path in to_process:
        score = blur_score(abs_path)
        if score is None:
            continue

        is_blurry = 1 if score < BLUR_THRESHOLD else 0
        ph = compute_phash(abs_path)
        is_duplicate = 0
        duplicate_of = None

        if ph and HAS_IMAGEHASH:
            ph_hash = imagehash.hex_to_hash(ph)
            for existing_path, existing_ph in existing_hashes.items():
                try:
                    if ph_hash - imagehash.hex_to_hash(existing_ph) < DUPE_THRESHOLD:
                        is_duplicate = 1
                        duplicate_of = existing_path
                        break
                except Exception:
                    pass
            if not is_duplicate:
                existing_hashes[rel_path] = ph

        conn.execute('''
            INSERT OR REPLACE INTO photo_ai
            (photo_path, blur_score, is_blurry, phash, is_duplicate, duplicate_of)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (rel_path, score, is_blurry, ph, is_duplicate, duplicate_of))

        counts['processed'] += 1
        if is_blurry:
            counts['blurry'] += 1
        if is_duplicate:
            counts['duplicates'] += 1

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
