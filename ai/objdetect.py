#!/usr/bin/env python3
"""
Viviti Object Detection — COCO-SSD MobileNet (TFLite).

Detects real objects (person, dog, cat, car, food, ...) with their own bundled
label map, so search terms map to accurate labels. Fully on-device, no network
at inference time.

Usage:
  python3 objdetect.py --download                 download model + labels
  python3 objdetect.py <image_path>               print detected labels (JSON)
  python3 objdetect.py --batch <photo_dir> <db>   detect across library, update DB

Requires: ai-edge-litert (or tflite-runtime), opencv-python, numpy
"""
import sys, os, json, zipfile, urllib.request
from pathlib import Path

MODELS_DIR  = Path(__file__).parent / 'models'
MODEL_ZIP_URL = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip'
MODEL_FILE  = 'coco_ssd_mobilenet_v1.tflite'
LABELS_FILE = 'coco_ssd_labels.txt'

SCORE_THRESHOLD = 0.40   # min detection confidence to keep
MAX_LABELS      = 8      # distinct labels stored per photo
IMAGE_EXT       = {'.jpg', '.jpeg', '.png', '.heic', '.cr2', '.arw', '.nef', '.dng'}


def ensure_assets():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path  = MODELS_DIR / MODEL_FILE
    labels_path = MODELS_DIR / LABELS_FILE
    if model_path.exists() and labels_path.exists():
        return str(model_path), str(labels_path)

    print('[objdetect] Downloading COCO-SSD MobileNet (~6.9 MB)...', file=sys.stderr)
    zip_path = MODELS_DIR / 'objdetect.zip'
    urllib.request.urlretrieve(MODEL_ZIP_URL, str(zip_path))
    with zipfile.ZipFile(str(zip_path)) as z:
        for member in z.namelist():
            if member.endswith('.tflite'):
                with z.open(member) as src, open(model_path, 'wb') as dst:
                    dst.write(src.read())
            elif 'labelmap' in member or member.endswith('labels.txt'):
                with z.open(member) as src, open(labels_path, 'wb') as dst:
                    dst.write(src.read())
    zip_path.unlink()
    return str(model_path), str(labels_path)


class ObjectDetector:
    """Loads the interpreter once; reuse across many images in one process."""
    def __init__(self):
        self._interp = None
        self._labels = None
        self._in_idx = None
        self._in_size = 300
        self._in_dtype = None
        self._out = None   # dict: boxes/classes/scores/count tensor indices

    def _load(self):
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            import tflite_runtime.interpreter as tflite
        model_path, labels_path = ensure_assets()
        self._interp = tflite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        self._in_idx   = inp['index']
        self._in_size  = int(inp['shape'][1])
        self._in_dtype = inp['dtype']

        with open(labels_path) as f:
            self._labels = [l.strip() for l in f if l.strip()]
        # This labelmap prepends a '???' placeholder, so the model's 0-based class
        # indices map to labels[idx + 1] (class 0 = "person"). Detect that and
        # offset; fall back to 0 for labelmaps without the placeholder.
        self._label_offset = 1 if self._labels and self._labels[0] in ('???', 'background', '') else 0

        # Map the 4 SSD-postprocess outputs by shape (order varies by export):
        #   boxes  -> ndim 3, last dim 4
        #   count  -> total size 1
        #   classes/scores -> [1, N]  (disambiguated after first inference)
        outs = self._interp.get_output_details()
        boxes_i = count_i = None
        flat = []
        for d in outs:
            shape = list(d['shape'])
            if len(shape) == 3 and shape[-1] == 4:
                boxes_i = d['index']
            elif int(np_prod(shape)) == 1:
                count_i = d['index']
            else:
                flat.append(d['index'])
        self._out = {'boxes': boxes_i, 'count': count_i, 'flat': flat,
                     'classes': None, 'scores': None}

    def detect(self, image_path):
        import cv2, numpy as np
        if self._interp is None:
            self._load()

        img = cv2.imread(image_path)
        if img is None:
            return []
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (self._in_size, self._in_size)).astype(self._in_dtype)
        self._interp.set_tensor(self._in_idx, inp[np.newaxis])
        self._interp.invoke()

        get = lambda i: self._interp.get_tensor(i)[0]
        a, b = get(self._out['flat'][0]), get(self._out['flat'][1])
        # Scores are confidences in [0,1]; classes are integer indices. Decide once.
        if self._out['classes'] is None:
            a_is_scores = float(np.max(a)) <= 1.0 and not _all_int(a)
            b_is_scores = float(np.max(b)) <= 1.0 and not _all_int(b)
            if a_is_scores and not b_is_scores:
                self._out['scores'], self._out['classes'] = self._out['flat'][0], self._out['flat'][1]
            elif b_is_scores and not a_is_scores:
                self._out['scores'], self._out['classes'] = self._out['flat'][1], self._out['flat'][0]
            else:
                # Fall back to documented order: classes first, scores second
                self._out['classes'], self._out['scores'] = self._out['flat'][0], self._out['flat'][1]

        classes = get(self._out['classes'])
        scores  = get(self._out['scores'])

        found = {}
        for cls, sc in zip(classes, scores):
            if sc < SCORE_THRESHOLD:
                continue
            idx = int(cls) + self._label_offset
            if 0 <= idx < len(self._labels):
                label = self._labels[idx].lower()
                if label in ('???', '', 'background'):
                    continue
                found[label] = max(found.get(label, 0.0), float(sc))
        # Highest-confidence labels first
        return [lbl for lbl, _ in sorted(found.items(), key=lambda kv: -kv[1])][:MAX_LABELS]


def np_prod(shape):
    p = 1
    for x in shape:
        p *= int(x)
    return p


def _all_int(arr):
    import numpy as np
    return bool(np.all(np.equal(np.mod(arr, 1), 0)))


_detector = None

def analyse(image_path):
    global _detector
    if _detector is None:
        _detector = ObjectDetector()
    return _detector.detect(image_path)


def run_batch(photo_dir, db_path):
    import sqlite3
    det = ObjectDetector()
    conn = sqlite3.connect(db_path)
    # objects column already exists (created by batch.py); ensure table is there
    conn.execute('''CREATE TABLE IF NOT EXISTS photo_ai (
        photo_path TEXT PRIMARY KEY, objects TEXT )''')

    updated = 0
    for root, _, files in os.walk(photo_dir):
        for f in files:
            if Path(f).suffix.lower() not in IMAGE_EXT:
                continue
            abs_path = os.path.join(root, f)
            rel_path = os.path.relpath(abs_path, photo_dir).replace('\\', '/')
            try:
                labels = det.detect(abs_path)
            except Exception as e:
                print(f'[objdetect] {rel_path}: {e}', file=sys.stderr)
                continue
            conn.execute(
                '''INSERT INTO photo_ai (photo_path, objects) VALUES (?, ?)
                   ON CONFLICT(photo_path) DO UPDATE SET objects=excluded.objects''',
                (rel_path, json.dumps(labels)),
            )
            updated += 1
            if updated % 25 == 0:
                conn.commit()
    conn.commit()

    # Prune orphan rows whose photo files were deleted — otherwise stale labels
    # linger and search can return dead (blank) results.
    removed = 0
    for (pp,) in conn.execute('SELECT photo_path FROM photo_ai').fetchall():
        ap = os.path.join(photo_dir, pp.replace('/', os.sep))
        if not os.path.exists(ap):
            conn.execute('DELETE FROM photo_ai WHERE photo_path=?', (pp,))
            removed += 1
    conn.commit()
    conn.close()
    print(json.dumps({'updated': updated, 'pruned_orphans': removed}))


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--download':
        ensure_assets()
        print(json.dumps({'ok': True}))
    elif len(sys.argv) == 4 and sys.argv[1] == '--batch':
        run_batch(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 2:
        print(json.dumps({'labels': analyse(sys.argv[1])}))
    else:
        print(json.dumps({'error': 'Usage: objdetect.py --download | <image> | --batch <dir> <db>'}))
        sys.exit(1)
