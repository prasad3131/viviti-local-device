#!/bin/bash
# Download face detection + recognition models.
# Run once on the Pi: bash /opt/viviti/ai/setup_models.sh
set -e
DIR="$(dirname "$0")/models"
mkdir -p "$DIR"

# ── YuNet face detector (234 KB) ─────────────────────────────────────────────
YUNET="$DIR/face_detection_yunet_2023mar.onnx"
if [ ! -f "$YUNET" ]; then
  echo "Downloading YuNet face detection model (~234 KB)..."
  wget -q --show-progress -O "$YUNET" \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fi
echo "YuNet ready: $YUNET"

# ── SFace face recognizer — int8 quantized (9.3 MB) ──────────────────────────
# OpenCV FaceRecognizerSF: 128-d embeddings, much more accurate than histograms
SFACE="$DIR/face_recognition_sface_2021dec_int8.onnx"
if [ ! -f "$SFACE" ]; then
  echo "Downloading SFace face recognition model (~9.3 MB)..."
  wget -q --show-progress -O "$SFACE" \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx"
fi
echo "SFace ready: $SFACE"

# ── COCO-SSD MobileNet object detector — TFLite quant (~6.9 MB) ───────────────
# Bundled labelmap (no class-index guessing). Powers accurate Object Search.
OBJ="$DIR/coco_ssd_mobilenet_v1.tflite"
if [ ! -f "$OBJ" ]; then
  echo "Downloading COCO-SSD object detection model (~6.9 MB)..."
  TMP="$DIR/objdetect.zip"
  wget -q --show-progress -O "$TMP" \
    "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
  unzip -o -j "$TMP" '*.tflite' -d "$DIR" >/dev/null
  unzip -o -j "$TMP" '*labelmap*' -d "$DIR" >/dev/null
  # Normalize filenames to what objdetect.py expects
  [ -f "$DIR/detect.tflite" ] && mv -f "$DIR/detect.tflite" "$OBJ"
  [ -f "$DIR/labelmap.txt" ] && mv -f "$DIR/labelmap.txt" "$DIR/coco_ssd_labels.txt"
  rm -f "$TMP"
fi
echo "Object detector ready: $OBJ"

echo ""
echo "All models downloaded. Restart viviti: sudo systemctl restart viviti"
