#!/usr/bin/env python3
"""
Viviti Face Detection — OpenCV DNN ResNet-SSD detector + histogram clustering.
Much more accurate than Haar cascade: handles angled faces, partial occlusion,
varied lighting, and rejects non-face false positives via confidence threshold.

Run setup_models.sh first to download the model files (~10 MB).
Usage: python3 faces.py <photo_dir> <db_path>
"""
import sys, os, json, sqlite3, hashlib
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
CONFIDENCE_THRESHOLD = 0.35   # Ignore detections below 35% confidence
MIN_FACE_PX = 30              # Ignore faces smaller than 30px (noise)
SIMILARITY_THRESHOLD = 0.14   # Cosine distance threshold for clustering

MODEL_DIR = Path(__file__).parent / 'models'
PROTO_PATH = MODEL_DIR / 'deploy.prototxt'
MODEL_PATH = MODEL_DIR / 'res10_300x300_ssd.caffemodel'


def _load_net():
    if not PROTO_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'DNN model files missing. Run: bash /opt/viviti/ai/setup_models.sh'
        )
    net = cv2.dnn.readNetFromCaffe(str(PROTO_PATH), str(MODEL_PATH))
    # Use CPU — Orange Pi has no CUDA
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


NET = _load_net()


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

    # DNN expects 300x300 blob
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    NET.setInput(blob)
    detections = NET.forward()  # shape: (1, 1, N, 7)

    results = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Bounding box in absolute pixels
        box = detections[0, 0, i, 3:7] * np.array([iw, ih, iw, ih])
        x1, y1, x2, y2 = box.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(iw, x2), min(ih, y2)
        fw, fh = x2 - x1, y2 - y1

        if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
            continue

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        hist = face_histogram(crop)
        key = hashlib.md5(f'{img_path}{x1}{y1}'.encode()).hexdigest()[:10]
        thumb_path = os.path.join(thumb_dir, f'face_{key}.jpg')
        cv2.imwrite(thumb_path, cv2.resize(crop, (120, 120)))
        results.append({
            'x': x1, 'y': y1, 'w': fw, 'h': fh,
            'histogram': hist, 'thumb_path': thumb_path,
        })
    return results


def assign_cluster(conn, histogram, exclude=None):
    if exclude is None:
        exclude = set()
    arr = np.array(histogram)
    rows = conn.execute(
        'SELECT id, centroid FROM face_clusters WHERE centroid IS NOT NULL'
    ).fetchall()
    best_id, best_dist = None, float('inf')
    for cid, cent_json in rows:
        if cid in exclude:
            continue
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

    done_faces = set(
        (r[0], r[1], r[2])
        for r in conn.execute('SELECT photo_path, x, y FROM photo_faces')
    )
    counts = {'processed': 0, 'faces': 0}

    for root, _, files in os.walk(photo_dir):
        for f in files:
            if Path(f).suffix.lower() not in IMAGE_EXT:
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, photo_dir).replace('\\', '/')
            face_list = detect_faces_in(abs_path, thumb_dir)
            if not face_list:
                counts['processed'] += 1
                continue
            used_this_photo = set()
            for face in face_list:
                if (rel_path, face['x'], face['y']) in done_faces:
                    continue
                cid = assign_cluster(conn, face['histogram'], exclude=used_this_photo)
                used_this_photo.add(cid)
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
