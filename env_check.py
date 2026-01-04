import cv2
import torch
import pyttsx3
from mediapipe.tasks.python.vision import HandLandmarker
from mediapipe.tasks.python.core import base_options

print("OpenCV:", cv2.__version__)
print("Torch:", torch.__version__)
print("MPS:", torch.backends.mps.is_available())
print("MediaPipe Tasks OK")
print("Environment ready")
