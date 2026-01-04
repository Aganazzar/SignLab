#!/usr/bin/env python3
"""
Download MediaPipe Face Landmarker model
"""
import os
import urllib.request

FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
FACE_MODEL_PATH = "mediapipe/face_landmarker.task"

os.makedirs("mediapipe", exist_ok=True)

if os.path.exists(FACE_MODEL_PATH):
    print(f"✓ Face landmarker model already exists: {FACE_MODEL_PATH}")
else:
    print(f"📥 Downloading face landmarker model...")
    print(f"   URL: {FACE_MODEL_URL}")
    print(f"   Destination: {FACE_MODEL_PATH}")
    
    try:
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        size_mb = os.path.getsize(FACE_MODEL_PATH) / (1024 * 1024)
        print(f"✓ Downloaded successfully ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"   Please download manually from: {FACE_MODEL_URL}")
        print(f"   Save it to: {FACE_MODEL_PATH}")

# Check hand model
HAND_MODEL_PATH = "mediapipe/hand_landmarker.task"
if os.path.exists(HAND_MODEL_PATH):
    print(f"✓ Hand landmarker model exists: {HAND_MODEL_PATH}")
else:
    print(f"⚠️  Hand landmarker model not found: {HAND_MODEL_PATH}")
    print(f"   Make sure you have the hand model file!")

print("\n" + "="*60)
print("Setup complete! You can now:")
print("  1. python collect_data.py  (collect training data)")
print("  2. python train.py         (train the model)")
print("  3. python -m realtime.realtime_inference  (run inference)")
print("="*60)
