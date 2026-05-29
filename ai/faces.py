#!/usr/bin/env python3
"""
Viviti Face Detection — Haar cascade + histogram clustering.
No additional installs required (uses opencv-python).
Usage: python3 faces.py <photo_dir> <db_path>
"""
import sys, os, json, sqlite3, hashlib
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
SIMILARITY_THRESHOLD = 0.14

def _find_cascade():
    candidates = [
        str(Path(__file__).parent / 'models' / 'haarcascade_frontalface_default.xml'),
        '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
        '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
    ]
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        candidates.insert(0, cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError('haarcascade_frontalface_default.xml not found')

FACE_CASCADE = cv2.CascadeClassifier(_find_cascade())


def face_histogram(face_bgr):
    hsv = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], None, [64], [0, 256]).flatten()
    feat = np.concatenate([h_hist, s_hist, v_hist])
    norm = np.linalg.norm(feat)
    return (feat / norm).tolist() if norm > 0 else feat.tolist()


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b))


def init_db(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS face_clusters (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT,
            centroid    TEXT,
            sample_thumb TEXT,
            photo_count INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime("now"))
        )
    ''')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS photo_faces (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_path  TEXT NOT NULL,
            cluster_id  INTEGER REFERENCES face_clusters(id),
            histogram   TEXT NOT NULL,
            x INTEGER, y INTEGER, w INTEGER, h INTEGER,
            thumb_path  TEXT
        )
    ''')
    try:
        conn.execute('CREATE INDEX IF NOT EXISTS idx_pf_cluster ON photo_faces(cluster_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_pf_photo ON photo_faces(photo_path)')
    except Exception:
        pass
    conn.commit()


def detect_faces_in(img_path, thumb_dir):
    img = cv2.imread(img_path)
    if img is None:
        return []
    ih, iw = img.shape[:2]
    scale = min(1.0, 800 / max(ih, iw))
    small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                       (int(iw * scale), int(ih * scale)))
    rects = FACE_CASCADE.detectMultiScale(
        small, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)
    )
    results = []
    if not len(rects):
        return results
    for (x, y, fw, fh) in rects:
        ox, oy = int(x / scale), int(y / scale)
        ow, oh = int(fw / scale), int(fh / scale)
        pad = int(oh * 0.18)
        x1, y1 = max(0, ox - pad), max(0, oy - pad)
        x2, y2 = min(iw, ox + ow + pad), min(ih, oy + oh + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        hist = face_histogram(crop)
        key = hashlib.md5(f'{img_path}{ox}{oy}'.encode()).hexdigest()[:10]
        thumb_path = os.path.join(thumb_dir, f'face_{key}.jpg')
        cv2.imwrite(thumb_path, cv2.resize(crop, (120, 120)))
        results.append({'x': ox, 'y': oy, 'w': ow, 'h': oh,
                        'histogram': hist, 'thumb_path': thumb_path})
    return results


def assign_cluster(conn, histogram):
    arr = np.array(histogram)
    rows = conn.execute(
        'SELECT id, centroid FROM face_clusters WHERE centroid IS NOT NULL'
    ).fetchall()
    best_id, best_dist = None, float('inf')
    for cid, cent_json in rows:
        d = cosine_dist(arr, np.array(json.loads(cent_json)))
        if d < best_dist:
            best_dist, best_id = d, cid
    if best_id is not None and best_dist < SIMILARITY_THRESHOLD:
        old = np.array(json.loads(
            conn.execute('SELECT centroid FROM face_clusters WHERE id=?', (best_id,)).fetchone()[0]
        ))
        new_c = old * 0.8 + arr * 0.2
        norm = np.linalg.norm(new_c)
        if norm > 0:
            new_c /= norm
        conn.execute(
            'UPDATE face_clusters SET centroid=?, photo_count=photo_count+1 WHERE id=?',
            (json.dumps(new_c.tolist()), best_id)
        )
        return best_id
    cur = conn.execute(
        'INSERT INTO face_clusters (centroid, photo_count) VALUES (?, 1)',
        (json.dumps(histogram),)
    )
    return cur.lastrowid


def run_faces(photo_dir, db_path):
    thumb_dir = os.path.join(os.path.dirname(db_path), 'face_thumbs')
    os.makedirs(thumb_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    done = {r[0] for r in conn.execute('SELECT DISTINCT photo_path FROM photo_faces')}
    counts = {'processed': 0, 'faces': 0}

    for root, _, files in os.walk(photo_dir):
        for f in files:
            if Path(f).suffix.lower() not in IMAGE_EXT:
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, photo_dir).replace('\\', '/')
            if rel_path in done:
                continue
            for face in detect_faces_in(abs_path, thumb_dir):
                cid = assign_cluster(conn, face['histogram'])
                existing_thumb = conn.execute(
                    'SELECT sample_thumb FROM face_clusters WHERE id=?', (cid,)
                ).fetchone()
                if existing_thumb and not existing_thumb[0]:
                    conn.execute(
                        'UPDATE face_clusters SET sample_thumb=? WHERE id=?',
                        (face['thumb_path'], cid)
                    )
                conn.execute(
                    'INSERT INTO photo_faces (photo_path,cluster_id,histogram,x,y,w,h,thumb_path) '
                    'VALUES (?,?,?,?,?,?,?,?)',
                    (rel_path, cid, json.dumps(face['histogram']),
                     face['x'], face['y'], face['w'], face['h'], face['thumb_path'])
                )
                counts['faces'] += 1
            counts['processed'] += 1
            if counts['processed'] % 20 == 0:
                conn.commit()

    conn.commit()
    conn.close()
    print(json.dumps(counts))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: faces.py <photo_dir> <db_path>'}))
        sys.exit(1)
    run_faces(sys.argv[1], sys.argv[2])
