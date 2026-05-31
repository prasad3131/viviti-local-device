#!/usr/bin/env python3
"""
Viviti Face Detection — YuNet (cv2.FaceDetectorYN) + histogram clustering.
YuNet handles non-frontal, partially occluded, and downward-looking faces —
ideal for casual party/family photos. Requires OpenCV 4.5+ (Pi has 4.10.0).

Run setup_models.sh first to download the model (~234 KB).
Usage: python3 faces.py <photo_dir> <db_path>
"""
import sys, os, json, sqlite3, hashlib
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
SCORE_THRESHOLD       = 0.13   # YuNet confidence (downward-looking party faces score 0.13-0.2)
NMS_THRESHOLD         = 0.3    # Non-maximum suppression
MIN_FACE_PX           = 30     # Ignore faces smaller than 30px
MAX_FACE_AR           = 1.1    # Skip detections wider than tall — real faces are portrait
SIMILARITY_THRESHOLD  = 0.22   # Merge into existing cluster if cosine dist < this
CONSOLIDATION_THRESHOLD = 0.18 # Merge two existing clusters if their centroids are this close

MODEL_DIR  = Path(__file__).parent / 'models'
MODEL_PATH = MODEL_DIR / 'face_detection_yunet_2023mar.onnx'
MAX_DIM    = 1280         # Pre-scale long edge to this before detection


def _load_detector():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f'YuNet model missing. Run: bash /opt/viviti/ai/setup_models.sh'
        )
    return cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (320, 320),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=5000,
    )


DETECTOR = _load_detector()


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
            best_score  REAL DEFAULT 0,
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
            score       REAL DEFAULT 0,
            x INTEGER, y INTEGER, w INTEGER, h INTEGER,
            thumb_path  TEXT
        )
    ''')
    # Add new columns to existing tables (safe to run multiple times)
    for sql in [
        'ALTER TABLE face_clusters ADD COLUMN best_score REAL DEFAULT 0',
        'ALTER TABLE photo_faces ADD COLUMN score REAL DEFAULT 0',
    ]:
        try:
            conn.execute(sql)
        except Exception:
            pass
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

    scale = min(1.0, MAX_DIM / max(ih, iw))
    work = cv2.resize(img, (int(iw * scale), int(ih * scale))) if scale < 1.0 else img
    wh, ww = work.shape[:2]

    DETECTOR.setInputSize((ww, wh))
    _, faces = DETECTOR.detect(work)

    if faces is None:
        return []

    results = []
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        score = float(face[14])

        x1 = max(0, int(x / scale))
        y1 = max(0, int(y / scale))
        fw = int(w / scale)
        fh = int(h / scale)
        x2 = min(iw, x1 + fw)
        y2 = min(ih, y1 + fh)
        fw, fh = x2 - x1, y2 - y1

        if fw < MIN_FACE_PX or fh < MIN_FACE_PX:
            continue
        if fh > 0 and fw / fh > MAX_FACE_AR:
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
            'histogram': hist, 'thumb_path': thumb_path, 'score': score,
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


def consolidate_clusters(conn):
    """Merge clusters whose centroids are within CONSOLIDATION_THRESHOLD.
    Runs after all photos are processed to collapse duplicate clusters of the same person.
    The cluster with more photos survives; named clusters take priority."""
    rows = conn.execute(
        'SELECT id, centroid, name, photo_count, sample_thumb, best_score '
        'FROM face_clusters WHERE centroid IS NOT NULL ORDER BY photo_count DESC'
    ).fetchall()
    merged = set()

    for i in range(len(rows)):
        id1, c1, name1, count1, thumb1, score1 = rows[i]
        if id1 in merged:
            continue
        arr1 = np.array(json.loads(c1))

        for j in range(i + 1, len(rows)):
            id2, c2, name2, count2, thumb2, score2 = rows[j]
            if id2 in merged:
                continue
            d = cosine_dist(arr1, np.array(json.loads(c2)))
            if d < CONSOLIDATION_THRESHOLD:
                # Merge id2 into id1
                conn.execute('UPDATE photo_faces SET cluster_id=? WHERE cluster_id=?', (id1, id2))
                conn.execute('UPDATE face_clusters SET photo_count=photo_count+? WHERE id=?', (count2, id1))
                # Adopt id2's name if id1 is unnamed
                if name2 and not name1:
                    conn.execute('UPDATE face_clusters SET name=? WHERE id=?', (name2, id1))
                    name1 = name2
                # Adopt id2's thumbnail if it has a better score
                if (score2 or 0) > (score1 or 0):
                    conn.execute('UPDATE face_clusters SET sample_thumb=?, best_score=? WHERE id=?',
                                 (thumb2, score2, id1))
                    score1, thumb1 = score2, thumb2
                conn.execute('DELETE FROM face_clusters WHERE id=?', (id2,))
                merged.add(id2)

    if merged:
        conn.commit()


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
                # Update sample_thumb if this face has a higher score (more frontal)
                row = conn.execute(
                    'SELECT sample_thumb, best_score FROM face_clusters WHERE id=?', (cid,)
                ).fetchone()
                if row and (not row[0] or face['score'] > (row[1] or 0)):
                    conn.execute(
                        'UPDATE face_clusters SET sample_thumb=?, best_score=? WHERE id=?',
                        (face['thumb_path'], face['score'], cid)
                    )
                conn.execute(
                    'INSERT INTO photo_faces (photo_path,cluster_id,histogram,score,x,y,w,h,thumb_path) '
                    'VALUES (?,?,?,?,?,?,?,?,?)',
                    (rel_path, cid, json.dumps(face['histogram']), face['score'],
                     face['x'], face['y'], face['w'], face['h'], face['thumb_path'])
                )
                counts['faces'] += 1
            counts['processed'] += 1
            if counts['processed'] % 20 == 0:
                conn.commit()

    conn.commit()
    consolidate_clusters(conn)
    conn.close()
    print(json.dumps(counts))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(json.dumps({'error': 'Usage: faces.py <photo_dir> <db_path>'}))
        sys.exit(1)
    run_faces(sys.argv[1], sys.argv[2])
