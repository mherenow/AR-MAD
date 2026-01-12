import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_rgb_histogram(image_path: str, output_path: str | None):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)  # H x W x 3

    colors = [("Red", "red", 0), ("Green", "green", 1), ("Blue", "blue", 2)]
    x = np.arange(256)

    plt.figure(figsize=(8, 5), dpi=120)
    for label, color, ch in colors:
        hist, _ = np.histogram(arr[:, :, ch].ravel(), bins=256, range=(0, 256))
        plt.plot(x, hist, color=color, label=label, linewidth=1)

    plt.xlim(0, 255)
    plt.xlabel("Pixel value (0–255)")
    plt.ylabel("Frequency")
    plt.title("Color Distribution (RGB)")
    plt.legend()
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
        plt.savefig(output_path)
        print(f"Saved histogram to: {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Plot RGB color distribution histogram of an image.")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--output", "-o", help="Path to save the histogram image (e.g., out.png). If omitted, a window is shown.")
    args = parser.parse_args()

    plot_rgb_histogram(args.image, args.output)

if __name__ == "__main__":
    main()
