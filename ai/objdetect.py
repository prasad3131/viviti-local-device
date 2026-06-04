#!/usr/bin/env python3
"""
Viviti Object Detection — EfficientDet-Lite0 (TFLite).

Detects real objects (person, dog, cat, car, food, ...) far more accurately than
the old SSD MobileNet v1 (which confidently mislabelled e.g. a rose as a carrot).
Labels are read straight from the model's embedded metadata, so class indices
can't mismatch. Fully on-device, no network at inference.

Usage:
  python3 objdetect.py --download                 download model
  python3 objdetect.py <image_path>               print detected labels (JSON)
  python3 objdetect.py --batch <photo_dir> <db>   detect across library, update DB

Requires: ai-edge-litert (or tflite-runtime), opencv-python, numpy
"""
import sys, os, json, zipfile, urllib.request
from pathlib import Path

MODELS_DIR  = Path(__file__).parent / 'models'
# EfficientDet-Lite0, int8, with embedded label metadata (~4.5 MB).
MODEL_URL   = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite/task_library/object_detection/android/lite-model_efficientdet_lite0_detection_metadata_1.tflite'
MODEL_FILE  = 'efficientdet_lite0.tflite'

SCORE_THRESHOLD = 0.40   # min detection confidence to keep
MAX_LABELS      = 8      # distinct labels stored per photo
IMAGE_EXT       = {'.jpg', '.jpeg', '.png', '.heic', '.cr2', '.arw', '.nef', '.dng'}

# Fallback only — used if the model has no embedded labelmap (it should).
COCO80 = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


def ensure_assets():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / MODEL_FILE
    if not model_path.exists():
        print('[objdetect] Downloading EfficientDet-Lite0 (~4.5 MB)...', file=sys.stderr)
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    return str(model_path)


def _extract_labels(model_path):
    """TFLite metadata models append their associated files (labelmap) as a zip
    archive to the .tflite. Read it directly — no separate file, no guessing."""
    try:
        with zipfile.ZipFile(model_path) as z:
            for n in z.namelist():
                if 'label' in n.lower() or n.endswith('.txt'):
                    return [l.strip() for l in z.read(n).decode('utf-8').splitlines() if l.strip()]
    except Exception:
        pass
    return None


class ObjectDetector:
    """Loads the interpreter once; reuse across many images in one process."""
    def __init__(self):
        self._interp = None
        self._labels = None
        self._label_offset = 0
        self._in_idx = None
        self._in_size = 320
        self._in_dtype = None
        self._out = None

    def _load(self):
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            import tflite_runtime.interpreter as tflite
        model_path = ensure_assets()
        self._interp = tflite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        self._in_idx   = inp['index']
        self._in_size  = int(inp['shape'][1])
        self._in_dtype = inp['dtype']

        self._labels = _extract_labels(model_path) or COCO80
        # Some labelmaps prepend a '???'/'background' placeholder, shifting indices.
        self._label_offset = 1 if self._labels and self._labels[0] in ('???', 'background', '') else 0

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
        inp = cv2.resize(rgb, (self._in_size, self._in_size))
        if np.issubdtype(self._in_dtype, np.floating):
            inp = inp.astype(self._in_dtype) / 255.0    # float models expect [0,1]
        else:
            inp = inp.astype(self._in_dtype)             # quantized uint8 [0,255]
        self._interp.set_tensor(self._in_idx, inp[np.newaxis])
        self._interp.invoke()

        get = lambda i: self._interp.get_tensor(i)[0]
        a, b = get(self._out['flat'][0]), get(self._out['flat'][1])
        if self._out['classes'] is None:
            a_is_scores = float(np.max(a)) <= 1.0 and not _all_int(a)
            b_is_scores = float(np.max(b)) <= 1.0 and not _all_int(b)
            if a_is_scores and not b_is_scores:
                self._out['scores'], self._out['classes'] = self._out['flat'][0], self._out['flat'][1]
            elif b_is_scores and not a_is_scores:
                self._out['scores'], self._out['classes'] = self._out['flat'][1], self._out['flat'][0]
            else:
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

    # Prune orphan rows whose photo files were deleted.
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
