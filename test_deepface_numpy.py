from deepface import DeepFace
import cv2
import numpy as np
import os

# Create a dummy image
img = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)

dataset_path = "dataset_faces"

try:
    # Test if DeepFace.find accepts numpy array
    print("Testing DeepFace.find with numpy array...")
    dfs = DeepFace.find(img_path=img, db_path=dataset_path, enforce_detection=False, silent=True)
    print("Success: DeepFace.find accepted numpy array.")
except Exception as e:
    print(f"Failed: DeepFace.find does not accept numpy array directly (or other error). Error: {e}")
