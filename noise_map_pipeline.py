import cv2
import numpy as np
import pywt
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
from typing import Tuple

def read_image(path: str, as_gray: bool = True) -> np.ndarray:
    """Read image as float32 in range [0,1]."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if as_gray:
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return img

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Scale a float image (may contain negative values) to uint8 for display."""
    mn, mx = np.min(img), np.max(img)
    if mx == mn:
        return np.zeros_like(img, dtype=np.uint8)
    scaled = 255 * (img - mn) / (mx - mn)
    return scaled.astype(np.uint8)

def gaussian_hp_residual(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian high-pass: residual = img - GaussianBlur(img). Works on grayscale floats [0,1]."""
    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    ksize = int(2 * round(3 * sigma) + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
    residual = img - blurred
    return residual

def wavelet_residual(img: np.ndarray, wavelet: str = 'db2', level: int = 1) -> np.ndarray:
    """Extract high-frequency components using discrete wavelet transform and reconstruct high-pass residual.
       Works on 2D grayscale float images in [0,1].
    """
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    # Zero-out approximation coeff (low-frequency) to keep only details
    coeffs_HP = list(coeffs)
    coeffs_HP[0] = np.zeros_like(coeffs_HP[0])
    highpass = pywt.waverec2(coeffs_HP, wavelet=wavelet)
    # crop to original shape (waverec may change shape slightly)
    highpass = highpass[:img.shape[0], :img.shape[1]]
    return highpass

def nlmeans_residual(img: np.ndarray, patch_size: int = 5, patch_distance: int = 6, h: float = None) -> np.ndarray:
    """Denoise using scikit-image's non-local means, then subtract to get residual.
       Input should be grayscale float in [0,1].
    """
    sigma_est = np.mean(estimate_sigma(img, multichannel=False))
    if h is None:
        h = 0.8 * sigma_est  # heuristic
    den = denoise_nl_means(img, h=h, patch_size=patch_size, patch_distance=patch_distance, preserve_range=True, fast_mode=True)
    residual = img - den
    return residual

def extract_noise_map(img: np.ndarray,
                      method: str = 'gaussian_hp',
                      normalize: bool = False,
                      **kwargs) -> np.ndarray:
    """
    Main wrapper. Returns float residual (can be negative centered around ~0).
    If normalize=True, returns uint8 normalized map for display.
    """
    if img.ndim == 3:
        # convert to grayscale luminance using standard Rec.709 weights
        img_gray = 0.2126 * img[...,0] + 0.7152 * img[...,1] + 0.0722 * img[...,2]
    else:
        img_gray = img

    if method == 'gaussian_hp':
        residual = gaussian_hp_residual(img_gray, sigma=kwargs.get('sigma', 1.0))
    elif method == 'wavelet':
        residual = wavelet_residual(img_gray, wavelet=kwargs.get('wavelet','db2'), level=kwargs.get('level',1))
    elif method == 'nlmeans':
        residual = nlmeans_residual(img_gray,
                                    patch_size=kwargs.get('patch_size',5),
                                    patch_distance=kwargs.get('patch_distance',6),
                                    h=kwargs.get('h', None))
    else:
        raise ValueError(f"Unknown method: {method}")

    if normalize:
        return normalize_to_uint8(residual)
    return residual

def visualize_maps(orig: np.ndarray, residual: np.ndarray, cmap: str = 'seismic'):
    """Quick visualization: original (grayscale) and residual heatmap."""
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    if orig.ndim == 3:
        plt.imshow(orig)
    else:
        plt.imshow(orig, cmap='gray')
    plt.title('Original (or grayscale)')
    plt.axis('off')

    plt.subplot(1,2,2)
    # residual often has positive and negative; use diverging colormap
    plt.imshow(residual, cmap=cmap)
    plt.title('Noise / Residual (high-freq)')
    plt.colorbar(shrink=0.7)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage when run as script:
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Input image file")
    parser.add_argument("--method", choices=['gaussian_hp','wavelet','nlmeans'], default='gaussian_hp')
    parser.add_argument("--sigma", type=float, default=1.0, help="sigma for gaussian_hp")
    args = parser.parse_args()

    img = read_image(args.image_path, as_gray=False)  # keep color, will be converted to gray internally
    residual = extract_noise_map(img, method=args.method, sigma=args.sigma, normalize=False)
    visualize_maps(img if img.ndim==2 else cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2GRAY), residual)