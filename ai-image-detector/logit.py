"""
Logit/Probability Analysis for AI Image Detector

Loads 100 fake and 100 real images (50 RAISE + 50 COCO2017),
runs inference with the all_features checkpoint, visualizes the
output probabilities per image, and finds the optimal decision threshold.

Note: The model's ClassificationHead ends with Sigmoid, so the output
is a probability in [0, 1] (1 = fake, 0 = real). We treat these as
"logit-equivalent" scores for threshold analysis.
"""

import os
import sys
import random
import warnings
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, auc

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── constants ─────────────────────────────────────────────────────────────────
CHECKPOINT   = ROOT.parent / "checkpoints" / "all_features" / "checkpoint_epoch_25.pth"
SYNTHBUSTER  = ROOT.parent / "datasets" / "synthbuster"
COCO_ROOT    = ROOT.parent / "datasets" / "coco2017" / "train2017"
N_PER_CLASS  = 1000          # total real = 100, total fake = 100
N_REAL_RAISE = N_PER_CLASS // 2   # 50 from RAISE
N_REAL_COCO  = N_PER_CLASS // 2   # 50 from COCO2017
SEED         = 67
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── model loading (reuse classify_image.py logic) ─────────────────────────────

def load_model(checkpoint_path: Path):
    import torch.nn as nn
    from models.classifier import BinaryClassifier

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Auto-detect architecture from state dict
    use_spectral       = any("spectral_branch"        in k for k in state_dict)
    use_noise_imprint  = any("noise_branch"            in k for k in state_dict)
    use_color_features = any("chrominance_branch"      in k for k in state_dict)
    use_local_patches  = any("local_patch_classifier"  in k for k in state_dict)
    use_fpn            = any("fpn"                     in k for k in state_dict)
    enable_attribution = any("attribution_head"        in k for k in state_dict)

    use_attention = None
    if any("attention_module.channel_attention" in k for k in state_dict):
        use_attention = "cbam"
    elif any("attention_module.fc1" in k for k in state_dict):
        use_attention = "se"

    clf_key = "classifier.classifier.0.weight"
    if clf_key in state_dict:
        dim = state_dict[clf_key].shape[1]
        backbone_type = "resnet50" if dim >= 2048 else "resnet18"
    else:
        backbone_type = "resnet18"

    model = BinaryClassifier(
        backbone_type=backbone_type,
        pretrained=False,
        use_spectral=use_spectral,
        use_noise_imprint=use_noise_imprint,
        use_color_features=use_color_features,
        use_local_patches=use_local_patches,
        use_fpn=use_fpn,
        use_attention=use_attention,
        enable_attribution=enable_attribution,
    )

    # Pre-register lazy pos_embedding for SpectralPatchTokenizer
    for name, param in state_dict.items():
        if name.endswith(".pos_embedding"):
            parts = name.split(".")
            module = model
            for part in parts[:-1]:
                module = getattr(module, part)
            if getattr(module, "pos_embedding", None) is None:
                module.pos_embedding = nn.Parameter(torch.empty_like(param))
                module.num_patches = param.shape[1]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model


# ── image collection ──────────────────────────────────────────────────────────

def collect_images(paths: list[Path], n: int, label: int, source: str) -> list[dict]:
    """Sample n images from a list of paths."""
    random.seed(SEED)
    chosen = random.sample(paths, min(n, len(paths)))
    return [{"path": p, "label": label, "source": source} for p in chosen]


def gather_samples() -> list[dict]:
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []

    # ── RAISE real images ──────────────────────────────────────────────────
    raise_dir = SYNTHBUSTER / "RAISE"
    if not raise_dir.exists():
        warnings.warn(f"RAISE directory not found: {raise_dir}")
    else:
        raise_paths = [p for p in raise_dir.iterdir() if p.suffix.lower() in valid_ext]
        if len(raise_paths) < N_REAL_RAISE:
            warnings.warn(f"Only {len(raise_paths)} RAISE images found, need {N_REAL_RAISE}")
        samples += collect_images(raise_paths, N_REAL_RAISE, label=0, source="RAISE")

    # ── COCO2017 real images ───────────────────────────────────────────────
    if not COCO_ROOT.exists():
        warnings.warn(f"COCO2017 directory not found: {COCO_ROOT}")
    else:
        coco_paths = [p for p in COCO_ROOT.iterdir() if p.suffix.lower() in valid_ext]
        if len(coco_paths) < N_REAL_COCO:
            warnings.warn(f"Only {len(coco_paths)} COCO images found, need {N_REAL_COCO}")
        samples += collect_images(coco_paths, N_REAL_COCO, label=0, source="COCO2017")

    # ── Fake images (all non-RAISE subdirs) ───────────────────────────────
    fake_paths = []
    if SYNTHBUSTER.exists():
        for gen_dir in SYNTHBUSTER.iterdir():
            if gen_dir.is_dir() and gen_dir.name != "RAISE":
                fake_paths += [p for p in gen_dir.iterdir() if p.suffix.lower() in valid_ext]
    if len(fake_paths) < N_PER_CLASS:
        warnings.warn(f"Only {len(fake_paths)} fake images found, need {N_PER_CLASS}")
    samples += collect_images(fake_paths, N_PER_CLASS, label=1, source="Fake")

    print(f"Collected {sum(s['label']==0 for s in samples)} real, "
          f"{sum(s['label']==1 for s in samples)} fake images.")
    return samples


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model, samples: list[dict]) -> list[dict]:
    scores = []
    with torch.no_grad():
        for i, s in enumerate(samples):
            try:
                img = Image.open(s["path"]).convert("RGB")
                tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
                out = model(tensor)
                # Handle (prediction, attribution) tuple
                if isinstance(out, tuple):
                    out = out[0]
                prob = out.squeeze().item()   # sigmoid output in [0,1]
                scores.append({**s, "score": prob})
            except Exception as e:
                warnings.warn(f"Skipping {s['path']}: {e}")
    return scores


# ── threshold search ──────────────────────────────────────────────────────────

def find_optimal_threshold(scores: list[dict]) -> float:
    """Youden's J statistic: maximises TPR - FPR."""
    y_true  = np.array([s["label"] for s in scores])
    y_score = np.array([s["score"] for s in scores])
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx]), fpr, tpr, thresholds


# ── visualisation ─────────────────────────────────────────────────────────────

def visualize(scores: list[dict], threshold: float, fpr, tpr, thresholds):
    real_scores = [s["score"] for s in scores if s["label"] == 0]
    fake_scores = [s["score"] for s in scores if s["label"] == 1]
    raise_scores = [s["score"] for s in scores if s["source"] == "RAISE"]
    coco_scores  = [s["score"] for s in scores if s["source"] == "COCO2017"]

    avg_real = np.mean(real_scores)
    avg_fake = np.mean(fake_scores)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("AI Image Detector – Probability Score Analysis\n"
                 f"(checkpoint_epoch_25.pth  |  threshold = {threshold:.4f})",
                 fontsize=14, fontweight="bold")

    # ── 1. Scatter: all individual scores ─────────────────────────────────
    ax = axes[0, 0]
    x_real = range(len(real_scores))
    x_fake = range(len(fake_scores))
    ax.scatter(x_real, real_scores, color="steelblue", alpha=0.6, s=18, label="Real")
    ax.scatter(x_fake, fake_scores, color="tomato",    alpha=0.6, s=18, label="Fake")
    ax.axhline(avg_real, color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Avg Real = {avg_real:.4f}")
    ax.axhline(avg_fake, color="tomato",    linestyle="--", linewidth=1.5,
               label=f"Avg Fake = {avg_fake:.4f}")
    ax.axhline(threshold, color="black", linestyle="-", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_title("Individual Scores (Real vs Fake)")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Model score (probability)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # ── 2. Histogram overlay ──────────────────────────────────────────────
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 40)
    ax.hist(real_scores, bins=bins, alpha=0.6, color="steelblue", label="Real")
    ax.hist(fake_scores, bins=bins, alpha=0.6, color="tomato",    label="Fake")
    ax.axvline(avg_real,   color="steelblue", linestyle="--", linewidth=1.5,
               label=f"Avg Real = {avg_real:.4f}")
    ax.axvline(avg_fake,   color="tomato",    linestyle="--", linewidth=1.5,
               label=f"Avg Fake = {avg_fake:.4f}")
    ax.axvline(threshold,  color="black",     linestyle="-",  linewidth=2,
               label=f"Threshold = {threshold:.4f}")
    ax.set_title("Score Distribution")
    ax.set_xlabel("Model score (probability)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # ── 3. Real sub-sources: RAISE vs COCO ────────────────────────────────
    ax = axes[1, 0]
    ax.scatter(range(len(raise_scores)), raise_scores, color="mediumseagreen",
               alpha=0.7, s=18, label=f"RAISE (n={len(raise_scores)})")
    ax.scatter(range(len(coco_scores)),  coco_scores,  color="darkorange",
               alpha=0.7, s=18, label=f"COCO2017 (n={len(coco_scores)})")
    ax.axhline(np.mean(raise_scores) if raise_scores else 0,
               color="mediumseagreen", linestyle="--", linewidth=1.5,
               label=f"Avg RAISE = {np.mean(raise_scores):.4f}" if raise_scores else "")
    ax.axhline(np.mean(coco_scores) if coco_scores else 0,
               color="darkorange", linestyle="--", linewidth=1.5,
               label=f"Avg COCO = {np.mean(coco_scores):.4f}" if coco_scores else "")
    ax.axhline(threshold, color="black", linestyle="-", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")
    ax.set_title("Real Sub-sources: RAISE vs COCO2017")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Model score (probability)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)

    # ── 4. ROC curve ──────────────────────────────────────────────────────
    ax = axes[1, 1]
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color="purple", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    # Mark optimal threshold point
    y_true  = np.array([s["label"] for s in scores])
    y_score = np.array([s["score"] for s in scores])
    fpr_opt = fpr[np.argmin(np.abs(thresholds - threshold))]
    tpr_opt = tpr[np.argmin(np.abs(thresholds - threshold))]
    ax.scatter([fpr_opt], [tpr_opt], color="red", zorder=5, s=80,
               label=f"Optimal threshold = {threshold:.4f}")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = ROOT / "logit_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out_path}")
    plt.close()

    # ── summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  SCORE SUMMARY")
    print("=" * 55)
    print(f"  Real  (n={len(real_scores):3d})  avg={avg_real:.4f}  "
          f"min={min(real_scores):.4f}  max={max(real_scores):.4f}")
    if raise_scores:
        print(f"    ↳ RAISE   (n={len(raise_scores):3d})  avg={np.mean(raise_scores):.4f}")
    if coco_scores:
        print(f"    ↳ COCO    (n={len(coco_scores):3d})  avg={np.mean(coco_scores):.4f}")
    print(f"  Fake  (n={len(fake_scores):3d})  avg={avg_fake:.4f}  "
          f"min={min(fake_scores):.4f}  max={max(fake_scores):.4f}")
    print("-" * 55)
    print(f"  Optimal threshold (Youden's J): {threshold:.4f}")
    print(f"  ROC AUC: {auc(fpr, tpr):.4f}")
    print("=" * 55)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model   = load_model(CHECKPOINT)
    samples = gather_samples()

    if not samples:
        raise RuntimeError("No samples collected – check dataset paths.")

    print(f"Running inference on {len(samples)} images (device={DEVICE})…")
    scores = run_inference(model, samples)

    threshold, fpr, tpr, thresholds = find_optimal_threshold(scores)
    visualize(scores, threshold, fpr, tpr, thresholds)


if __name__ == "__main__":
    main()
