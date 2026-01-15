"""
Command to run:
python -m pytest -q tests/test_00_env_imports.py
"""
#

def test_imports():
    import cv2
    import numpy
    import pycocotools
    import sklearn
    import torch

    assert hasattr(cv2, "imread")
    assert hasattr(torch, "cuda")

def test_cuda_optional():
    import torch
    # Not required to be True on every machine, but should not crash
    _ = torch.cuda.is_available()
