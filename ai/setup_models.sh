#!/bin/bash
# Download YuNet face detection model (~234 KB)
# Requires OpenCV 4.5+ (cv2.FaceDetectorYN) — Orange Pi has 4.10.0
# Run once on the Pi: bash /opt/viviti/ai/setup_models.sh
set -e
DIR="$(dirname "$0")/models"
mkdir -p "$DIR"

MODEL="$DIR/face_detection_yunet_2023mar.onnx"

if [ ! -f "$MODEL" ]; then
  echo "Downloading YuNet face detection model (~234 KB)..."
  wget -q --show-progress -O "$MODEL" \
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
fi

echo "YuNet model ready: $MODEL"
