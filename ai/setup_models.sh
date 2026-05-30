#!/bin/bash
# Download OpenCV DNN face detection model files (~10 MB total)
# Run once on the Pi: bash /opt/viviti/ai/setup_models.sh
set -e
DIR="$(dirname "$0")/models"
mkdir -p "$DIR"

PROTO="$DIR/deploy.prototxt"
MODEL="$DIR/res10_300x300_ssd.caffemodel"

if [ ! -f "$PROTO" ]; then
  echo "Downloading deploy.prototxt..."
  wget -q -O "$PROTO" \
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
fi

if [ ! -f "$MODEL" ]; then
  echo "Downloading face detection model (~10 MB)..."
  wget -q --show-progress -O "$MODEL" \
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
fi

echo "Models ready in $DIR"
