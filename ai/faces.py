#!/usr/bin/env python3
"""
Viviti Face Detection + Recognition

Detection:   YuNet (cv2.FaceDetectorYN) — fast, handles partial faces
Recognition: SFace (cv2.FaceRecognizerSF) — 128-d neural embedding, ONNX
             Falls back to LBP+HSV if SFace model not downloaded yet.

Run setup once:  bash /opt/viviti/ai/setup_models.sh
Usage:           python3 faces.py <photo_dir> <db_path>
"""
import sys, os, json, sqlite3, hashlib
import cv2
import numpy as np
from pathlib import Path

IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
MIN_FACE_PX      = 40    # raised — very small detections are usually noise
MAX_FACE_AR      = 1.15  # tighter — real faces are nearly square or portrait
MIN_FACE_AREA_RATIO = 0.003  # face must be ≥0.3% of image area (filters background tiny faces)

# Thresholds when using SFace embeddings (128-d, cosine distance)
SFACE_SIMILARITY_THRESHOLD    = 0.50   # merge into cluster if cosine dist < this
SFACE_NAMED_THRESHOLD         = 0.35   # stricter for named clusters
SFACE_CONSOLIDATION_THRESHOLD = 0.40

# Thresholds when using LBP fallback
LBP_SIMILARITY_THRESHOLD      = 0.18
LBP_NAMED_THRESHOLD           = 0.10
LBP_CONSOLIDATION_THRESHOLD   = 0.14

MODEL_DIR  = Path(__file__).parent / 'models'
YUNET_PATH = MODEL_DIR / 'face_detection_yunet_2023mar.onnx'
SFACE_PATH = MODEL_DIR / 'face_recognition_sface_2021dec_int8.onnx'
MAX_DIM    = 1280

SCORE_THRESHOLD = 0.45   # raised from 0.20 — SFace handles ID, detection can be strict
NMS_THRESHOLD   = 0.3


# ── Load models ───────────────────────────────────────────────────────────────

def _load_yunet():
    if not YUNET_PATH.exists():
        raise FileNotFoundError(f'Run: bash /opt/viviti/ai/setup_models.sh')
    return cv2.FaceDetectorYN.create(
        str(YUNET_PATH), '', (320, 320),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=NMS_THRESHOLD,
        top_k=5000,
    )

def _load_sface():
    if not SFACE_PATH.exists():
        return None
    try:
        return cv2.FaceRecognizerSF.create(str(SFACE_PATH), '')
    except Exception:
        return None

DETECTOR  = _load_yunet()
RECOGNIZER = _load_sface()
USE_SFACE  = RECOGNIZER is not None

SIMILARITY_THRESHOLD    = SFACE_SIMILARITY_THRESHOLD    if USE_SFACE else LBP_SIMILARITY_THRESHOLD
NAMED_THRESHOLD         = SFACE_NAMED_THRESHOLD         if USE_SFACE else LBP_NAMED_THRESHOLD
CONSOLIDATION_THRESHOLD = SFACE_CONSOLIDATION_THRESHOLD if USE_SFACE else LBP_CONSOLIDATION_THRESHOLD


# ── Feature extraction ────────────────────────────────────────────────────────

def _sface_embedding(img_bgr, yunet_face_row):
    """128-d L2-normalised embedding via SFace alignCrop."""
    aligned = RECOGNIZER.alignCrop(img_bgr, yunet_face_row)
    feat    = RECOGNIZER.feature(aligned).flatten()
    norm    = np.linalg.norm(feat)
    return (feat / norm).tolist() if norm > 0 else feat.tolist()


def _lbp_embedding(face_bgr):
    """LBP texture + HSV colour descriptor (fallback when SFace unavailable)."""
    gray    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (64, 64))
    lbp_hist = np.zeros(256, dtype=np.float32)
    rows, cols = resized.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            center = int(resized[r, c])
            code = 0
            neighbours = [
                resized[r-1,c-1], resized[r-1,c], resized[r-1,c+1],
                resized[r,  c+1],
                resized[r+1,c+1], resized[r+1,c], resized[r+1,c-1],
                resized[r,  c-1],
            ]
            for i, nb in enumerate(neighbours):
                if int(nb) >= center:
                    code |= (1 << i)
            lbp_hist[code] += 1
    hsv    = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
    feat   = np.concatenate([lbp_hist * 2.0, h_hist, s_hist])
    norm   = np.linalg.norm(feat)
    return (feat / norm).tolist() if norm > 0 else feat.tolist()


def cosine_dist(a, b):
    return 1.0 - float(np.dot(a, b))


# ── Post-detection quality filters ────────────────────────────────────────────

def has_face_texture(face_bgr):
    small = cv2.resize(face_bgr, (40, 40))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() > 12


def has_skin_tone(face_bgr, min_ratio=0.18):  # raised — dark false positives fail this
    hsv  = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0,  20, 40), (18, 180, 240)),
        cv2.inRange(hsv, (170, 20, 40), (180, 180, 240)),
    )
    return (np.count_nonzero(mask) / mask.size) >= min_ratio


def _yunet_valid_structure(face_row):
    x, y, w, h = face_row[:4].astype(float)
    re, le = face_row[4:6], face_row[6:8]
    nos    = face_row[8:10]
    rm, lm = face_row[10:12], face_row[12:14]
    pad = max(w, h) * 0.25
    for pt in (re, le, nos, rm, lm):
        if not (x-pad <= pt[0] <= x+w+pad and y-pad <= pt[1] <= y+h+pad):
            return False
    eye_y   = (re[1] + le[1]) / 2.0
    mouth_y = (rm[1] + lm[1]) / 2.0
    if not (eye_y < nos[1] < mouth_y): return False
    eye_sep = abs(re[0] - le[0])
    if eye_sep < w * 0.20: return False
    if not (0.08*h <= nos[1]-eye_y   <= 0.50*h): return False
    if not (0.06*h <= mouth_y-nos[1] <= 0.45*h): return False
    if not (0.20*h <= mouth_y-eye_y  <= 0.70*h): return False
    if not (0.20*w <= eye_sep        <= 0.80*w): return False
    return True


# ── Main detection ─────────────────────────────────────────────────────────────

def detect_faces_in(img_path, thumb_dir, img=None):
    # img_path is also used as the thumb-key seed; pass a pre-loaded frame (img)
    # to detect on a video frame without re-reading from disk.
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return []
    ih, iw = img.shape[:2]

    scale = min(1.0, MAX_DIM / max(ih, iw))
    work  = cv2.resize(img, (int(iw*scale), int(ih*scale))) if scale < 1.0 else img
    wh, ww = work.shape[:2]

    DETECTOR.setInputSize((ww, wh))
    _, faces = DETECTOR.detect(work)
    if faces is None:
        return []

    results = []
    for face in faces:
        if not _yunet_valid_structure(face):
            continue

        x, y, w, h = face[:4].astype(int)
        score = float(face[14])

        x1 = max(0, int(x / scale));  y1 = max(0, int(y / scale))
        fw  = int(w / scale);          fh = int(h / scale)
        x2  = min(iw, x1+fw);         y2 = min(ih, y1+fh)
        fw, fh = x2-x1, y2-y1

        if fw < MIN_FACE_PX or fh < MIN_FACE_PX: continue
        if fh > 0 and fw/fh > MAX_FACE_AR:        continue
        if (fw * fh) / (iw * ih) < MIN_FACE_AREA_RATIO: continue  # skip tiny background faces

        crop = img[y1:y2, x1:x2]
        if crop.size == 0:             continue
        if not has_skin_tone(crop):    continue
        if not has_face_texture(crop): continue

        # ── Embedding ────────────────────────────────────────────────────────
        if USE_SFACE:
            # Scale the YuNet detection row back to work-image coordinates
            # (alignCrop operates on the work image, not the original)
            work_face = face.copy()
            embedding = _sface_embedding(work, work_face)
        else:
            embedding = _lbp_embedding(crop)

        key        = hashlib.md5(f'{img_path}{x1}{y1}'.encode()).hexdigest()[:10]
        thumb_path = os.path.join(thumb_dir, f'face_{key}.jpg')
        cv2.imwrite(thumb_path, cv2.resize(crop, (120, 120)))

        results.append({
            'x': x1, 'y': y1, 'w': fw, 'h': fh,
            'embedding': embedding, 'thumb_path': thumb_path, 'score': score,
        })
    return results


# ── Database ──────────────────────────────────────────────────────────────────

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
    for sql in [
        'ALTER TABLE face_clusters ADD COLUMN best_score REAL DEFAULT 0',
        'ALTER TABLE photo_faces ADD COLUMN score REAL DEFAULT 0',
    ]:
        try: conn.execute(sql)
        except Exception: pass
    try:
        conn.execute('CREATE INDEX IF NOT EXISTS idx_pf_cluster ON photo_faces(cluster_id)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_pf_photo ON photo_faces(photo_path)')
    except Exception: pass
    conn.commit()


def assign_cluster(conn, embedding, exclude=None):
    if exclude is None:
        exclude = set()
    arr  = np.array(embedding)
    rows = conn.execute(
        'SELECT id, centroid, name FROM face_clusters WHERE centroid IS NOT NULL'
    ).fetchall()
    best_id, best_dist = None, float('inf')
    for cid, cent_json, name in rows:
        if cid in exclude: continue
        d = cosine_dist(arr, np.array(json.loads(cent_json)))
        thresh = NAMED_THRESHOLD if name else SIMILARITY_THRESHOLD
        if d < best_dist and d < thresh:
            best_dist, best_id = d, cid
    if best_id is not None:
        old   = np.array(json.loads(
            conn.execute('SELECT centroid FROM face_clusters WHERE id=?', (best_id,)).fetchone()[0]
        ))
        new_c = old * 0.85 + arr * 0.15
        norm  = np.linalg.norm(new_c)
        if norm > 0: new_c /= norm
        conn.execute(
            'UPDATE face_clusters SET centroid=?, photo_count=photo_count+1 WHERE id=?',
            (json.dumps(new_c.tolist()), best_id)
        )
        return best_id
    cur = conn.execute(
        'INSERT INTO face_clusters (centroid, photo_count) VALUES (?, 1)',
        (json.dumps(embedding),)
    )
    return cur.lastrowid


def consolidate_clusters(conn):
    """Only merge UNNAMED clusters — named clusters are permanently locked."""
    rows = conn.execute(
        'SELECT id, centroid, name, photo_count, sample_thumb, best_score '
        'FROM face_clusters WHERE centroid IS NOT NULL ORDER BY photo_count DESC'
    ).fetchall()
    merged = set()
    for i in range(len(rows)):
        id1, c1, name1, count1, thumb1, score1 = rows[i]
        if id1 in merged or name1: continue   # locked
        arr1 = np.array(json.loads(c1))
        for j in range(i+1, len(rows)):
            id2, c2, name2, count2, thumb2, score2 = rows[j]
            if id2 in merged or name2: continue  # locked
            if cosine_dist(arr1, np.array(json.loads(c2))) < CONSOLIDATION_THRESHOLD:
                conn.execute('UPDATE photo_faces SET cluster_id=? WHERE cluster_id=?', (id1, id2))
                conn.execute('UPDATE face_clusters SET photo_count=photo_count+? WHERE id=?', (count2, id1))
                if (score2 or 0) > (score1 or 0):
                    conn.execute('UPDATE face_clusters SET sample_thumb=?, best_score=? WHERE id=?',
                                 (thumb2, score2, id1))
                    score1, thumb1 = score2, thumb2
                conn.execute('DELETE FROM face_clusters WHERE id=?', (id2,))
                merged.add(id2)
    if merged: conn.commit()


def run_faces(photo_dir, db_path):
    thumb_dir = os.path.join(os.path.dirname(db_path), 'face_thumbs')
    os.makedirs(thumb_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    init_db(conn)

    done = set(
        (r[0], r[1], r[2])
        for r in conn.execute('SELECT photo_path, x, y FROM photo_faces')
    )
    counts = {'processed': 0, 'faces': 0,
              'engine': 'sface' if USE_SFACE else 'lbp'}

    for root, _, files in os.walk(photo_dir):
        for f in files:
            if Path(f).suffix.lower() not in IMAGE_EXT: continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, photo_dir).replace('\\', '/')
            face_list = detect_faces_in(abs_path, thumb_dir)
            if not face_list:
                counts['processed'] += 1
                continue
            used = set()
            for face in face_list:
                if (rel_path, face['x'], face['y']) in done: continue
                cid = assign_cluster(conn, face['embedding'], exclude=used)
                used.add(cid)
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
                    (rel_path, cid, json.dumps(face['embedding']), face['score'],
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
