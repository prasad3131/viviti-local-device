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

echo ""
echo "All models downloaded. Restart viviti: sudo systemctl restart viviti"
