import cv2
import numpy as np

# Generate a small image
img1 = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.rectangle(img1, (100, 100), (200, 200), (0, 255, 0), -1)
cv2.putText(img1, "Cam 1", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite("tests/fixtures/cam1.png", img1)

img2 = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.circle(img2, (320, 240), 100, (0, 0, 255), -1)
cv2.putText(img2, "Cam 2", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imwrite("tests/fixtures/cam2.png", img2)

print("Images generated in tests/fixtures/")
