import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_frequency_distribution(image_path):
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Compute 2D FFT and shift zero frequency to the center
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Compute magnitude spectrum (log scale for visibility)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return img, magnitude_spectrum

def visualize_frequency(image_path, title="Image"):
    img, magnitude_spectrum = get_frequency_distribution(image_path)

    # Plot original image and its frequency spectrum side-by-side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"{title} (Spatial Domain)")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(f"{title} (Frequency Domain)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage
# visualize_frequency("real_image.jpg", title="Real Image")
# visualize_frequency("ai_image.jpg", title="AI-Generated Image")
visualize_frequency("image.jpeg", title="Sample Image")