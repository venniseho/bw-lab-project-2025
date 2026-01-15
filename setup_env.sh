#!/usr/bin/env bash
set -e

echo "=== Setting up COCO + SAM environment ==="

# 1. Create Python environment
PYTHON_BIN=python3

$PYTHON_BIN -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# 2. Install Python dependencies
pip install \
    numpy \
    scipy \
    matplotlib \
    tqdm \
    pillow \
    scikit-image \
    scikit-learn \
    opencv-python \
    pycocotools \
    torchvision

# 3. Install Segment Anything
if [ ! -d "segment-anything" ]; then
    git clone https://github.com/facebookresearch/segment-anything.git
fi

pip install -e segment-anything

# 4. Create directory structure
mkdir -p COCO/annotations
mkdir -p COCO/val2014
mkdir -p checkpoints
mkdir -p outputs

echo "Directory structure created:"
echo "  COCO/annotations"
echo "  COCO/val2014"
echo "  checkpoints/"
echo "  outputs/"

# 5. Manual downloads
echo ""
echo "MANUAL STEPS REQUIRED"
echo "1. Download COCO val2014 images:"
echo "   https://cocodataset.org/#download"
echo "   Extract into: COCO/val2014/"
echo ""
echo "2. Download COCO annotations:"
echo "   instances_val2014.json"
echo "   Place into: COCO/annotations/"
echo ""
echo "3. Download SAM checkpoint:"
echo "   sam_vit_h_4b8939.pth"
echo "   Place into: checkpoints/"
echo ""
echo "Setup complete."
