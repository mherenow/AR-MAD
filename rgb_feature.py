import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image and convert to RGB
img = cv2.imread("image1.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)

# Compute vector magnitude and direction
magnitude = np.linalg.norm(img, axis=2)  # per-pixel vector magnitude
unit_vectors = img / (magnitude[..., np.newaxis] + 1e-8)  # normalized direction vectors

print("Magnitude map shape:", magnitude.shape)
print("RGB vector map shape:", unit_vectors.shape)

step = 20
Y, X = np.mgrid[0:img.shape[0]:step, 0:img.shape[1]:step]
U = unit_vectors[::step, ::step, 0]
V = unit_vectors[::step, ::step, 1]

plt.figure(figsize=(6,6))
plt.imshow(img.astype(np.uint8))
plt.quiver(X, Y, U, V, color='white', scale=20)
plt.title("RGB Vector Field (R-G components)")
plt.show()