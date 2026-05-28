#!/usr/bin/env python3
"""
Viviti Scene Detection — MobileNet V2 TFLite classification.
Usage: python3 scene.py <image_path>
       python3 scene.py --download    (download model + labels only)
Requires: pip3 install tflite-runtime
"""
import sys, json, tarfile, urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models'
MODEL_URL   = 'https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz'
MODEL_FILE  = 'mobilenet_v2_1.0_224_quant.tflite'
LABELS_URL  = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
LABELS_FILE = 'imagenet_labels.txt'

TAG_KEYWORDS = {
    'beach':     ['seashore', 'sandbar', 'dock', 'pier', 'lifeboat', 'catamaran', 'lakeside', 'shoal'],
    'food':      ['pizza', 'hotdog', 'cheeseburger', 'ice cream', 'cake', 'burrito', 'pretzel',
                  'bagel', 'plate', 'wine bottle', 'beer bottle', 'coffee mug', 'espresso'],
    'pets':      ['dog', 'cat', 'kitten', 'tabby', 'hamster', 'rabbit', 'parrot', 'goldfish',
                  'labrador', 'poodle', 'husky', 'golden retriever', 'bulldog', 'beagle', 'siamese'],
    'landscape': ['alp', 'valley', 'cliff', 'volcano', 'promontory', 'mountain', 'geyser',
                  'rainforest', 'forest', 'jungle'],
    'nature':    ['flower', 'mushroom', 'fern', 'butterfly', 'dragonfly', 'snail',
                  'daisy', 'sunflower', 'dandelion', 'bird'],
    'sport':     ['soccer', 'basketball', 'tennis', 'golf', 'ski', 'snowboard',
                  'football', 'baseball', 'racket', 'dumbbell', 'swimming'],
    'vehicle':   ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'airplane', 'ship',
                  'ambulance', 'minivan', 'sports car'],
    'birthday':  ['birthday cake', 'candle'],
    'indoor':    ['bedroom', 'library', 'bathroom', 'kitchen', 'bookcase', 'sofa',
                  'television', 'laptop', 'monitor', 'keyboard', 'wardrobe'],
    'outdoor':   ['park bench', 'fountain', 'garden', 'picket fence', 'stone wall'],
    'people':    ['person', 'man', 'woman', 'child', 'crowd', 'suit', 'jersey', 'face'],
}


def ensure_assets():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path  = MODELS_DIR / MODEL_FILE
    labels_path = MODELS_DIR / LABELS_FILE

    if not model_path.exists():
        print('[scene] Downloading MobileNet V2 (~4 MB)...', file=sys.stderr)
        archive_path = MODELS_DIR / 'model.tgz'
        urllib.request.urlretrieve(MODEL_URL, str(archive_path))
        with tarfile.open(str(archive_path)) as t:
            for member in t.getmembers():
                if member.name.endswith('.tflite'):
                    member.name = MODEL_FILE
                    t.extract(member, str(MODELS_DIR))
                    break
        archive_path.unlink()

    if not labels_path.exists():
        print('[scene] Downloading labels...', file=sys.stderr)
        urllib.request.urlretrieve(LABELS_URL, str(labels_path))

    return str(model_path), str(labels_path)


class SceneDetector:
    """Loads interpreter once; reuse across many images in the same process."""
    def __init__(self):
        self._interp = None
        self._labels = None
        self._in_idx = self._out_idx = self._dtype = None

    def _load(self):
        try:
            import ai_edge_litert.interpreter as tflite
        except ImportError:
            import tflite_runtime.interpreter as tflite
        model_path, labels_path = ensure_assets()
        self._interp = tflite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]
        self._in_idx  = inp['index']
        self._out_idx = out['index']
        self._dtype   = inp['dtype']
        with open(labels_path) as f:
            self._labels = [l.strip().lower() for l in f]

    def analyse(self, image_path):
        import cv2, numpy as np
        if self._interp is None:
            self._load()

        img = cv2.imread(image_path)
        if img is None:
            return {'scene_tags': [], 'objects': []}

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (224, 224)).astype(self._dtype)
        self._interp.set_tensor(self._in_idx, inp[np.newaxis])
        self._interp.invoke()
        output = self._interp.get_tensor(self._out_idx)[0]

        top_idx    = output.argsort()[-5:][::-1]
        top_labels = [(self._labels[i], int(output[i])) for i in top_idx if i < len(self._labels)]

        scene_tags, objects = [], []
        for label, score in top_labels:
            objects.append(label)
            if score < 10:
                continue
            for tag, keywords in TAG_KEYWORDS.items():
                if tag not in scene_tags and any(kw in label for kw in keywords):
                    scene_tags.append(tag)

        return {'scene_tags': scene_tags, 'objects': objects}


_detector = None

def analyse(image_path):
    global _detector
    if _detector is None:
        _detector = SceneDetector()
    return _detector.analyse(image_path)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--download':
        ensure_assets()
        print(json.dumps({'ok': True}))
    elif len(sys.argv) < 2:
        print(json.dumps({'error': 'Usage: scene.py <image_path>'}))
        sys.exit(1)
    else:
        print(json.dumps(analyse(sys.argv[1])))
