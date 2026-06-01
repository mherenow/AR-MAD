"""
multiple_classify.py
--------------------
Batch classification using the all_features checkpoint.

Modes (--mode):
  batch      [default]  Sample 100 fake + 50 RAISE + 50 COCO, run full
                        9-panel diagnostic plot + per-image console table.
  generator             Use ALL available images per source (every generator
                        folder, RAISE, COCO).  Reports accuracy / precision /
                        recall / F1 broken down by source and saves a
                        dedicated generator-accuracy plot.

Usage:
  python multiple_classify.py                   # batch mode
  python multiple_classify.py --mode generator  # generator accuracy mode
  python multiple_classify.py --mode batch --threshold 0.7
"""

import os
import sys
import random
import warnings
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    average_precision_score, precision_recall_curve,
)

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT      = Path(__file__).parent
DETECTOR     = PROJECT / "ai-image-detector"
CHECKPOINT   = PROJECT / "checkpoints" / "all_features" / "checkpoint_epoch_25.pth"
SYNTHBUSTER  = PROJECT / "datasets" / "synthbuster"
COCO_DIR     = PROJECT / "datasets" / "coco2017" / "train2017"

N_FAKE       = 1000
N_RAISE      = 500
N_COCO       = 500
SEED         = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD    = 0.9244          # decision boundary on model probability output

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ── model loading (mirrors classify_image.py, no double-sigmoid) ──────────────

def load_model():
    sys.path.insert(0, str(DETECTOR))
    import torch.nn as nn
    from models.classifier import BinaryClassifier

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    print(f"Loading: {CHECKPOINT}")
    ckpt       = torch.load(CHECKPOINT, map_location=DEVICE)
    state_dict = ckpt.get("model_state_dict", ckpt)

    use_spectral       = any("spectral_branch"       in k for k in state_dict)
    use_noise_imprint  = any("noise_branch"           in k for k in state_dict)
    use_color_features = any("chrominance_branch"     in k for k in state_dict)
    use_local_patches  = any("local_patch_classifier" in k for k in state_dict)
    use_fpn            = any(k.startswith("fpn")      for k in state_dict)
    enable_attribution = any("attribution_head"       in k for k in state_dict)

    use_attention = None
    if any("attention_module" in k for k in state_dict):
        use_attention = "cbam" if any("channel_attention" in k for k in state_dict) else "se"

    clf_key       = "classifier.classifier.0.weight"
    backbone_type = "resnet50" if (clf_key in state_dict and state_dict[clf_key].shape[1] >= 2048) else "resnet18"

    model = BinaryClassifier(
        backbone_type=backbone_type, pretrained=False,
        use_spectral=use_spectral, use_noise_imprint=use_noise_imprint,
        use_color_features=use_color_features, use_local_patches=use_local_patches,
        use_fpn=use_fpn, use_attention=use_attention, enable_attribution=enable_attribution,
    )

    for name, param in state_dict.items():
        if name.endswith(".pos_embedding"):
            parts, module = name.split("."), model
            for p in parts[:-1]:
                module = getattr(module, p)
            if getattr(module, "pos_embedding", None) is None:
                module.pos_embedding = nn.Parameter(torch.empty_like(param))
                module.num_patches   = param.shape[1]

    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"Model ready  [{backbone_type}, device={DEVICE}]")
    return model


# ── sample collection ─────────────────────────────────────────────────────────

def sample_paths(directory: Path, n: int) -> list[Path]:
    paths = [p for p in directory.iterdir() if p.suffix.lower() in VALID_EXT]
    if len(paths) < n:
        warnings.warn(f"Only {len(paths)} images in {directory}, requested {n}")
    random.seed(SEED)
    return random.sample(paths, min(n, len(paths)))


def gather_samples() -> list[dict]:
    samples = []

    # RAISE real
    raise_dir = SYNTHBUSTER / "RAISE"
    for p in sample_paths(raise_dir, N_RAISE):
        samples.append({"path": p, "label": 0, "source": "RAISE"})

    # COCO real
    for p in sample_paths(COCO_DIR, N_COCO):
        samples.append({"path": p, "label": 0, "source": "COCO2017"})

    # Fake (pool all generator folders, then sample)
    fake_pool: list[Path] = []
    for gen_dir in SYNTHBUSTER.iterdir():
        if gen_dir.is_dir() and gen_dir.name != "RAISE":
            fake_pool += [p for p in gen_dir.iterdir() if p.suffix.lower() in VALID_EXT]
    random.seed(SEED)
    for p in random.sample(fake_pool, min(N_FAKE, len(fake_pool))):
        samples.append({"path": p, "label": 1, "source": p.parent.name})

    n_real = sum(s["label"] == 0 for s in samples)
    n_fake = sum(s["label"] == 1 for s in samples)
    print(f"Samples  ->  real: {n_real}  fake: {n_fake}  total: {len(samples)}")
    return samples


# ── inference ─────────────────────────────────────────────────────────────────

def _load_one(sample: dict) -> tuple[dict, "torch.Tensor | None"]:
    """Load + transform one image in a worker thread."""
    try:
        img = Image.open(sample["path"]).convert("RGB")
        return sample, TRANSFORM(img)
    except Exception as e:
        warnings.warn(f"Skipping {sample['path'].name}: {e}")
        return sample, None


def run_inference(model, samples: list[dict], batch_size: int = 32) -> list[dict]:
    """
    Batched GPU inference with:
      - tqdm progress bar (per-image, shows throughput in img/s)
      - ThreadPoolExecutor: parallel image loading overlaps with GPU compute
      - non_blocking CUDA transfers + synchronize for correctness
    """
    results  : list[dict] = []
    skipped  : int        = 0
    use_cuda  = DEVICE.type == "cuda"

    loader = ThreadPoolExecutor(max_workers=4)

    bar = tqdm(
        total=len(samples),
        desc=f"Inference [{'GPU' if use_cuda else 'CPU'}]",
        unit="img",
        dynamic_ncols=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  [{elapsed}<{remaining}  {rate_fmt}]",
    )

    for start in range(0, len(samples), batch_size):
        batch_meta = samples[start : start + batch_size]

        # Load images in parallel threads while GPU runs the previous batch
        futures = [loader.submit(_load_one, s) for s in batch_meta]
        loaded  = [f.result() for f in futures]

        valid_meta    = [s for s, t in loaded if t is not None]
        valid_tensors = [t for s, t in loaded if t is not None]
        skipped      += sum(1 for _, t in loaded if t is None)

        if not valid_tensors:
            bar.update(len(batch_meta))
            continue

        batch_tensor = torch.stack(valid_tensors).to(DEVICE, non_blocking=use_cuda)

        with torch.no_grad():
            out = model(batch_tensor)
            if isinstance(out, tuple):
                out = out[0]
            if use_cuda:
                torch.cuda.synchronize()           # ensure scores are ready before .cpu()
            scores = out.squeeze(1).cpu().numpy()  # (B,)

        for s, score in zip(valid_meta, scores):
            results.append({**s, "score": float(score),
                             "pred": 1 if float(score) >= THRESHOLD else 0})

        bar.update(len(batch_meta))

    bar.close()
    loader.shutdown(wait=False)

    gpu_label = torch.cuda.get_device_name(DEVICE) if use_cuda else "CPU"
    tqdm.write(f"Done  {len(results)} classified, {skipped} skipped  [{gpu_label}]")
    return results


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[dict]) -> dict:
    y_true  = np.array([r["label"] for r in results])
    y_pred  = np.array([r["pred"]  for r in results])
    y_score = np.array([r["score"] for r in results])

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fpr, tpr, roc_thresh = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y_true, y_score)

    # threshold sweep for metrics curve (plot 7)
    sweep_t   = np.linspace(0.01, 0.99, 200)
    sweep_acc, sweep_prec, sweep_rec, sweep_f1 = [], [], [], []
    for t in sweep_t:
        yp = (y_score >= t).astype(int)
        sweep_acc.append(accuracy_score(y_true, yp))
        sweep_prec.append(precision_score(y_true, yp, zero_division=0))
        sweep_rec.append(recall_score(y_true, yp, zero_division=0))
        sweep_f1.append(f1_score(y_true, yp, zero_division=0))

    return {
        "accuracy":      accuracy_score(y_true, y_pred),
        "precision":     precision_score(y_true, y_pred, zero_division=0),
        "recall":        recall_score(y_true, y_pred, zero_division=0),
        "f1":            f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":       roc_auc_score(y_true, y_score),
        "avg_precision": average_precision_score(y_true, y_score),
        "specificity":   tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "fpr":           fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr":           fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "roc_fpr": fpr, "roc_tpr": tpr, "roc_thresh": roc_thresh,
        "pr_prec": pr_prec, "pr_rec": pr_rec, "pr_thresh": pr_thresh,
        "sweep_t": sweep_t, "sweep_acc": sweep_acc,
        "sweep_prec": sweep_prec, "sweep_rec": sweep_rec, "sweep_f1": sweep_f1,
        "cm": cm,
        "y_true": y_true, "y_score": y_score,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _shade_threshold(ax, threshold=THRESHOLD, orientation="h"):
    """Shade predicted-real (below) and predicted-fake (above) regions."""
    if orientation == "h":
        ax.axhspan(-0.05, threshold, color="steelblue", alpha=0.06, zorder=0)
        ax.axhspan(threshold, 1.05,  color="tomato",    alpha=0.06, zorder=0)
        ax.axhline(threshold, color="black", ls="--", lw=1.2, zorder=3)
    else:
        ax.axvspan(-0.05, threshold, color="steelblue", alpha=0.06, zorder=0)
        ax.axvspan(threshold, 1.05,  color="tomato",    alpha=0.06, zorder=0)
        ax.axvline(threshold, color="black", ls="--", lw=1.2, zorder=3,
                   label=f"Threshold={threshold}")


def _jitter(n, width=0.08):
    rng = np.random.default_rng(0)
    return rng.uniform(-width, width, n)


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_results(results: list[dict], m: dict):
    real_scores  = np.array([r["score"] for r in results if r["label"] == 0])
    fake_scores  = np.array([r["score"] for r in results if r["label"] == 1])
    raise_scores = np.array([r["score"] for r in results if r["source"] == "RAISE"])
    coco_scores  = np.array([r["score"] for r in results if r["source"] == "COCO2017"])

    gen_names  = sorted({r["source"] for r in results if r["label"] == 1})
    gen_scores = {g: np.array([r["score"] for r in results if r["source"] == g])
                  for g in gen_names}

    # error groups
    tp_scores = np.array([r["score"] for r in results if r["label"]==1 and r["pred"]==1])
    fp_scores = np.array([r["score"] for r in results if r["label"]==0 and r["pred"]==1])
    tn_scores = np.array([r["score"] for r in results if r["label"]==0 and r["pred"]==0])
    fn_scores = np.array([r["score"] for r in results if r["label"]==1 and r["pred"]==0])

    C_REAL, C_FAKE = "steelblue", "tomato"
    C_RAISE, C_COCO = "mediumseagreen", "darkorange"

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle(
        f"Batch Classification Results  |  checkpoint_epoch_25.pth  |  threshold={THRESHOLD}\n"
        f"Acc={m['accuracy']:.3f}  Prec={m['precision']:.3f}  Rec={m['recall']:.3f}  "
        f"F1={m['f1']:.3f}  ROC-AUC={m['roc_auc']:.3f}  AP={m['avg_precision']:.3f}",
        fontsize=12, fontweight="bold",
    )
    axes = fig.subplots(3, 3)

    # ── 1. Violin + jittered strip (sorted by score) ──────────────────────
    ax = axes[0, 0]
    df_vio = pd.DataFrame({
        "score": np.concatenate([real_scores, fake_scores]),
        "class": ["Real"] * len(real_scores) + ["Fake"] * len(fake_scores),
    })
    sns.violinplot(data=df_vio, x="class", y="score", hue="class",
                   palette={"Real": C_REAL, "Fake": C_FAKE},
                   inner=None, cut=0, ax=ax, order=["Real", "Fake"], alpha=0.55, legend=False)
    for i, (scores, col) in enumerate([(real_scores, C_REAL), (fake_scores, C_FAKE)]):
        sorted_s = np.sort(scores)
        ax.scatter(_jitter(len(sorted_s)) + i, sorted_s,
                   color=col, s=12, alpha=0.7, zorder=3)
    ax.axhline(THRESHOLD, color="black", ls="--", lw=1.4, label=f"Threshold={THRESHOLD}")
    ax.axhspan(-0.05, THRESHOLD, color=C_REAL, alpha=0.05)
    ax.axhspan(THRESHOLD, 1.05,  color=C_FAKE, alpha=0.05)
    ax.set(title="Score Distribution", ylabel="Score", ylim=(-0.05, 1.05))
    ax.legend(fontsize=7)

    # ── 2. KDE density curves ─────────────────────────────────────────────
    ax = axes[0, 1]
    xs = np.linspace(0, 1, 500)
    for scores, col, lbl in [(real_scores, C_REAL, "Real"), (fake_scores, C_FAKE, "Fake")]:
        if len(scores) > 1:
            kde = gaussian_kde(scores, bw_method=0.15)
            ys  = kde(xs)
            ax.plot(xs, ys, color=col, lw=2, label=lbl)
            ax.fill_between(xs, ys, alpha=0.18, color=col)
    _shade_threshold(ax, orientation="v")
    ax.axvline(np.mean(real_scores), color=C_REAL, ls=":", lw=1.3,
               label=f"μ real={np.mean(real_scores):.3f}")
    ax.axvline(np.mean(fake_scores), color=C_FAKE, ls=":", lw=1.3,
               label=f"μ fake={np.mean(fake_scores):.3f}")
    ax.set(title="KDE Score Density", xlabel="Score", ylabel="Density", xlim=(-0.02, 1.02))
    ax.legend(fontsize=7)

    # ── 3. ROC + PR curves ────────────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(m["roc_fpr"], m["roc_tpr"], color="purple", lw=2,
            label=f"ROC  AUC={m['roc_auc']:.4f}")
    ax.plot(m["pr_rec"], m["pr_prec"], color="darkcyan", lw=2,
            label=f"PR   AP={m['avg_precision']:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    # operating point on ROC
    idx = np.argmin(np.abs(m["roc_thresh"] - THRESHOLD))
    ax.scatter([m["roc_fpr"][idx]], [m["roc_tpr"][idx]],
               color="red", zorder=5, s=70, label=f"ROC @ t={THRESHOLD}")
    ax.set(title="ROC & Precision–Recall Curves", xlabel="FPR  /  Recall",
           ylabel="TPR  /  Precision", xlim=(-0.02, 1.02), ylim=(-0.02, 1.05))
    ax.legend(fontsize=7)

    # ── 4. Normalised confusion matrix ────────────────────────────────────
    ax = axes[1, 0]
    cm_raw  = m["cm"]
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    labels_xy = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            pct  = cm_norm[i, j]
            cnt  = cm_raw[i, j]
            tcol = "white" if pct > 0.55 else "black"
            ax.text(j, i, f"{pct:.1%}\n({cnt})", ha="center", va="center",
                    fontsize=11, color=tcol, fontweight="bold")
            ax.text(j, i + 0.38, labels_xy[i][j], ha="center", va="center",
                    fontsize=7, color=tcol, alpha=0.7)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Real", "Pred Fake"])
    ax.set_yticklabels(["True Real", "True Fake"])
    ax.set_title("Normalised Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # ── 5. Real sub-sources: side-by-side violin ──────────────────────────
    ax = axes[1, 1]
    df_src = pd.DataFrame({
        "score":  np.concatenate([raise_scores, coco_scores]),
        "source": ["RAISE"] * len(raise_scores) + ["COCO2017"] * len(coco_scores),
    })
    sns.violinplot(data=df_src, x="source", y="score", hue="source",
                   palette={"RAISE": C_RAISE, "COCO2017": C_COCO},
                   inner="box", cut=0, ax=ax, alpha=0.6, legend=False)
    for i, (scores, col) in enumerate([(raise_scores, C_RAISE), (coco_scores, C_COCO)]):
        ax.scatter(_jitter(len(scores)) + i, scores, color=col, s=12, alpha=0.65, zorder=3)
    ax.axhline(THRESHOLD, color="black", ls="--", lw=1.3, label=f"Threshold={THRESHOLD}")
    ax.axhspan(-0.05, THRESHOLD, color=C_REAL, alpha=0.05)
    ax.axhspan(THRESHOLD, 1.05,  color=C_FAKE, alpha=0.05)
    ax.set(title="Real Sub-sources (RAISE vs COCO)", ylabel="Score", ylim=(-0.05, 1.05))
    ax.legend(fontsize=7)

    # ── 6. Per-generator box + swarm ──────────────────────────────────────
    ax = axes[1, 2]
    tab_colors = plt.cm.tab10(np.linspace(0, 1, len(gen_names)))
    bp = ax.boxplot([gen_scores[g] for g in gen_names], patch_artist=True,
                    vert=True, widths=0.5, zorder=2)
    for patch, col in zip(bp["boxes"], tab_colors):
        patch.set_facecolor(col); patch.set_alpha(0.45)
    for i, (g, col) in enumerate(zip(gen_names, tab_colors)):
        jx = _jitter(len(gen_scores[g]), width=0.18) + (i + 1)
        ax.scatter(jx, gen_scores[g], color=col, s=14, alpha=0.75, zorder=3)
    _shade_threshold(ax, orientation="h")
    ax.set_xticks(range(1, len(gen_names) + 1))
    ax.set_xticklabels([g[:14] for g in gen_names], rotation=35, ha="right", fontsize=7)
    ax.set(title="Fake Scores by Generator",
           ylabel="Score", ylim=(-0.05, 1.05))

    # ── 7. Threshold vs Metrics curve ─────────────────────────────────────
    ax = axes[2, 0]
    ax.plot(m["sweep_t"], m["sweep_acc"],  color="royalblue",   lw=1.8, label="Accuracy")
    ax.plot(m["sweep_t"], m["sweep_prec"], color="darkorange",  lw=1.8, label="Precision")
    ax.plot(m["sweep_t"], m["sweep_rec"],  color="mediumseagreen", lw=1.8, label="Recall")
    ax.plot(m["sweep_t"], m["sweep_f1"],   color="purple",      lw=2.2, label="F1", ls="--")
    ax.axvline(THRESHOLD, color="black", ls="--", lw=1.3, label=f"t={THRESHOLD}")
    ax.axvspan(0, THRESHOLD, color="steelblue", alpha=0.04)
    ax.axvspan(THRESHOLD, 1, color="tomato",    alpha=0.04)
    ax.set(title="Threshold vs Metrics", xlabel="Threshold",
           ylabel="Score", xlim=(0, 1), ylim=(0, 1.05))
    ax.legend(fontsize=7)

    # ── 8. Error analysis: score distributions per TP/FP/TN/FN ───────────
    ax = axes[2, 1]
    groups = [
        (tp_scores, "TP (correct fake)",   "tomato",        "solid"),
        (fp_scores, "FP (real->fake)",       "lightsalmon",   "dashed"),
        (tn_scores, "TN (correct real)",    "steelblue",     "solid"),
        (fn_scores, "FN (fake->real)",       "lightsteelblue","dashed"),
    ]
    xs = np.linspace(0, 1, 400)
    for scores, lbl, col, ls in groups:
        if len(scores) > 1:
            kde = gaussian_kde(scores, bw_method=0.2)
            ys  = kde(xs)
            ax.plot(xs, ys, color=col, lw=2, ls=ls, label=f"{lbl} (n={len(scores)})")
            ax.fill_between(xs, ys, alpha=0.12, color=col)
        elif len(scores) == 1:
            ax.axvline(scores[0], color=col, lw=2, ls=ls,
                       label=f"{lbl} (n=1, s={scores[0]:.3f})")
    _shade_threshold(ax, orientation="v")
    ax.set(title="Error Analysis (KDE per TP/FP/TN/FN)",
           xlabel="Score", ylabel="Density", xlim=(-0.02, 1.02))
    ax.legend(fontsize=7)

    # ── 9. Sorted score strip (real + fake, sorted by score) ──────────────
    ax = axes[2, 2]
    all_sorted = sorted(results, key=lambda r: r["score"])
    xs_all     = np.arange(len(all_sorted))
    colors_all = [C_FAKE if r["label"] == 1 else C_REAL for r in all_sorted]
    edge_all   = ["black" if r["pred"] != r["label"] else "none" for r in all_sorted]
    ax.scatter(xs_all, [r["score"] for r in all_sorted],
               c=colors_all, edgecolors=edge_all, linewidths=0.8, s=16, alpha=0.8, zorder=3)
    _shade_threshold(ax, orientation="h")
    ax.set(title="All Scores Sorted  (* = misclassified)",
           xlabel="Rank (sorted by score)", ylabel="Score", ylim=(-0.05, 1.05))
    real_patch = mpatches.Patch(color=C_REAL, label="Real")
    fake_patch = mpatches.Patch(color=C_FAKE, label="Fake")
    err_patch  = mpatches.Patch(edgecolor="black", facecolor="white",
                                label="Misclassified", linewidth=1)
    ax.legend(handles=[real_patch, fake_patch, err_patch], fontsize=7)

    plt.tight_layout()
    out = PROJECT / "multiple_classify_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {out}")
    plt.close()


# ── console report ────────────────────────────────────────────────────────────

def print_report(results: list[dict], m: dict):
    print("\n" + "=" * 62)
    print("  PER-IMAGE RESULTS")
    print("=" * 62)
    print(f"  {'File':<35} {'Src':<14} {'Score':>6}  {'GT':>4}  {'Pred':>4}  {'OK'}")
    print("-" * 62)
    for r in results:
        ok   = "OK" if r["pred"] == r["label"] else "XX"
        gt   = "FAKE" if r["label"] == 1 else "REAL"
        pred = "FAKE" if r["pred"]  == 1 else "REAL"
        print(f"  {r['path'].name:<35} {r['source']:<14} {r['score']:>6.4f}  {gt:>4}  {pred:>4}  {ok}")

    print("\n" + "=" * 62)
    print("  CLASSIFICATION METRICS")
    print("=" * 62)
    print(f"  Accuracy          : {m['accuracy']:.4f}")
    print(f"  Precision         : {m['precision']:.4f}")
    print(f"  Recall (TPR)      : {m['recall']:.4f}")
    print(f"  Specificity (TNR) : {m['specificity']:.4f}")
    print(f"  F1 Score          : {m['f1']:.4f}")
    print(f"  ROC AUC           : {m['roc_auc']:.4f}")
    print(f"  Avg Precision     : {m['avg_precision']:.4f}")
    print(f"  FPR               : {m['fpr']:.4f}")
    print(f"  FNR               : {m['fnr']:.4f}")
    print("-" * 62)
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print("=" * 62)


# ── generator-accuracy mode ───────────────────────────────────────────────────

def gather_all_samples(max_per_source: int = 200) -> list[dict]:
    """
    Collect images from every source folder.
    Each source is capped at max_per_source images (default 200) so inference
    stays tractable. Use --max-per-source 1000 for a full evaluation.
    """
    samples = []
    rng     = random.Random(SEED)

    def _sample_dir(directory: Path, label: int, source: str):
        paths = [p for p in directory.iterdir() if p.suffix.lower() in VALID_EXT]
        chosen = rng.sample(paths, min(max_per_source, len(paths)))
        return [{"path": p, "label": label, "source": source} for p in chosen]

    # RAISE — real
    raise_dir = SYNTHBUSTER / "RAISE"
    if raise_dir.exists():
        samples += _sample_dir(raise_dir, 0, "RAISE")
    else:
        warnings.warn(f"RAISE directory not found: {raise_dir}")

    # COCO2017 — real
    if COCO_DIR.exists():
        samples += _sample_dir(COCO_DIR, 0, "COCO2017")
    else:
        warnings.warn(f"COCO directory not found: {COCO_DIR}")

    # All generator folders — fake
    if SYNTHBUSTER.exists():
        for gen_dir in sorted(SYNTHBUSTER.iterdir()):
            if gen_dir.is_dir() and gen_dir.name != "RAISE":
                samples += _sample_dir(gen_dir, 1, gen_dir.name)

    counts = {}
    for s in samples:
        counts[s["source"]] = counts.get(s["source"], 0) + 1
    print(f"\nSample counts (max {max_per_source} per source):")
    for src, n in sorted(counts.items()):
        tag = "real" if src in ("RAISE", "COCO2017") else "fake"
        print(f"  {src:<30} {n:>5}  [{tag}]")
    print(f"  {'TOTAL':<30} {len(samples):>5}")
    return samples


def compute_generator_metrics(results: list[dict]) -> list[dict]:
    """
    Per-source breakdown.

    For fake sources  → correct prediction = FAKE (label 1, pred 1)
    For real sources  → correct prediction = REAL (label 0, pred 0)
    Returns list of dicts sorted by accuracy descending.
    """
    sources = sorted({r["source"] for r in results})
    rows = []
    for src in sources:
        grp    = [r for r in results if r["source"] == src]
        label  = grp[0]["label"]          # all same within a source
        n      = len(grp)
        scores = np.array([r["score"] for r in grp])
        preds  = np.array([r["pred"]  for r in grp])
        labels = np.array([r["label"] for r in grp])

        correct = int((preds == labels).sum())
        acc     = correct / n

        if label == 1:   # fake source — TP / FN
            tp = int((preds == 1).sum())
            fn = n - tp
            fp = tn = 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:            # real source — TN / FP
            tn = int((preds == 0).sum())
            fp = n - tn
            tp = fn = 0
            precision = 1.0   # no fake predictions to evaluate precision on
            recall    = 0.0   # no positives in ground truth
            # use specificity as the meaningful metric for real sources
            acc = tn / n      # same as accuracy here

        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        rows.append({
            "source":    src,
            "type":      "real" if label == 0 else "fake",
            "n":         n,
            "correct":   correct,
            "accuracy":  acc,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "avg_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "scores":    scores,
        })

    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    return rows


def print_generator_report(rows: list[dict]):
    W = 110   # total table width

    # column widths
    C_SRC  = 28
    C_TYPE =  5
    C_N    =  6
    C_COR  =  7
    C_ACC  =  7
    C_PREC =  7
    C_REC  =  7
    C_F1   =  7
    C_AVG  =  8
    C_STD  =  7
    C_MIN  =  7
    C_MAX  =  7

    sep   = "=" * W
    dash  = "-" * W

    hdr = (f"  {'Source':<{C_SRC}} {'Type':<{C_TYPE}} {'N':>{C_N}}  {'Correct':>{C_COR}}  "
           f"{'Acc':>{C_ACC}}  {'Prec':>{C_PREC}}  {'Rec':>{C_REC}}  {'F1':>{C_F1}}  "
           f"{'AvgScore':>{C_AVG}}  {'Std':>{C_STD}}  {'Min':>{C_MIN}}  {'Max':>{C_MAX}}  Bar(20)")

    print("\n" + sep)
    print("  ACCURACY PER GENERATOR / SOURCE")
    print(sep)
    print(hdr)
    print(dash)

    for r in rows:
        bar_len = int(r["accuracy"] * 20)
        bar     = "#" * bar_len + "." * (20 - bar_len)
        mn      = float(r["scores"].min())
        mx      = float(r["scores"].max())
        print(
            f"  {r['source']:<{C_SRC}} {r['type']:<{C_TYPE}} {r['n']:>{C_N}}  "
            f"{r['correct']:>{C_COR}}  "
            f"{r['accuracy']:>{C_ACC}.4f}  "
            f"{r['precision']:>{C_PREC}.4f}  "
            f"{r['recall']:>{C_REC}.4f}  "
            f"{r['f1']:>{C_F1}.4f}  "
            f"{r['avg_score']:>{C_AVG}.4f}  "
            f"{r['std_score']:>{C_STD}.4f}  "
            f"{mn:>{C_MIN}.4f}  "
            f"{mx:>{C_MAX}.4f}  "
            f"[{bar}]"
        )

    # ── overall row ───────────────────────────────────────────────────────
    all_flat = []
    for r in rows:
        lbl = 0 if r["type"] == "real" else 1
        for s in r["scores"]:
            all_flat.append({"label": lbl, "pred": int(s >= THRESHOLD), "score": float(s)})

    y_true  = np.array([x["label"] for x in all_flat])
    y_pred  = np.array([x["pred"]  for x in all_flat])
    y_score = np.array([x["score"] for x in all_flat])

    overall_acc = accuracy_score(y_true, y_pred)
    overall_auc = roc_auc_score(y_true, y_score)
    n_correct   = int((y_true == y_pred).sum())

    print(dash)
    print(
        f"  {'OVERALL':<{C_SRC}} {'—':<{C_TYPE}} {len(y_true):>{C_N}}  "
        f"{n_correct:>{C_COR}}  "
        f"{overall_acc:>{C_ACC}.4f}  "
        f"{'—':>{C_PREC}}  {'—':>{C_REC}}  {'—':>{C_F1}}  "
        f"{float(y_score.mean()):>{C_AVG}.4f}  "
        f"{float(y_score.std()):>{C_STD}.4f}  "
        f"{float(y_score.min()):>{C_MIN}.4f}  "
        f"{float(y_score.max()):>{C_MAX}.4f}  "
        f"ROC-AUC={overall_auc:.4f}"
    )
    print(sep)
    print(f"  Threshold used: {THRESHOLD}  |  "
          f"Real sources: specificity reported as Acc  |  "
          f"Fake sources: recall reported as Acc")
    print(sep)


def plot_generator_accuracy(rows: list[dict]):
    """
    4-panel generator accuracy dashboard:
      1. Horizontal accuracy bar chart (all sources, colour-coded real/fake)
      2. Avg score per source with error bars (std) + threshold line
      3. Score distribution violin per source
      4. Heatmap: source × metric (Acc / Prec / Rec / F1)
    """
    C_REAL, C_FAKE = "steelblue", "tomato"
    labels   = [r["source"] for r in rows]
    accs     = [r["accuracy"]  for r in rows]
    avgs     = [r["avg_score"] for r in rows]
    stds     = [r["std_score"] for r in rows]
    bar_cols = [C_REAL if r["type"] == "real" else C_FAKE for r in rows]
    n_src    = len(rows)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(
        f"Generator / Source Accuracy  |  checkpoint_epoch_25.pth  |  threshold={THRESHOLD}",
        fontsize=13, fontweight="bold",
    )

    # ── 1. Accuracy bar chart ─────────────────────────────────────────────
    ax = axes[0, 0]
    y_pos = np.arange(n_src)
    bars  = ax.barh(y_pos, accs, color=bar_cols, alpha=0.80, edgecolor="white", height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.axvline(1.0, color="grey", ls=":", lw=0.8)
    for bar, acc, r in zip(bars, accs, rows):
        ax.text(min(acc + 0.01, 1.04), bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}  (n={r['n']})", va="center", fontsize=7.5)
    real_p = mpatches.Patch(color=C_REAL, label="Real source (specificity)")
    fake_p = mpatches.Patch(color=C_FAKE, label="Fake source (recall)")
    ax.legend(handles=[real_p, fake_p], fontsize=8, loc="lower right")
    ax.set(title="Accuracy per Source", xlabel="Accuracy")
    ax.invert_yaxis()

    # ── 2. Avg score ± std per source ─────────────────────────────────────
    ax = axes[0, 1]
    ax.barh(y_pos, avgs, xerr=stds, color=bar_cols, alpha=0.70,
            edgecolor="white", height=0.65, capsize=3)
    ax.axvline(THRESHOLD, color="black", ls="--", lw=1.4,
               label=f"Threshold={THRESHOLD}")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(-0.05, 1.1)
    ax.legend(fontsize=8)
    ax.set(title="Avg Score +/- Std per Source", xlabel="Score")
    ax.invert_yaxis()

    # ── 3. Score violin per source ────────────────────────────────────────
    ax = axes[1, 0]
    # build long-form dataframe
    df_rows = []
    for r in rows:
        for s in r["scores"]:
            df_rows.append({"source": r["source"], "score": float(s), "type": r["type"]})
    df = pd.DataFrame(df_rows)
    palette = {r["source"]: (C_REAL if r["type"] == "real" else C_FAKE) for r in rows}
    order   = [r["source"] for r in rows]
    sns.violinplot(data=df, x="score", y="source", hue="source",
                   palette=palette, order=order, inner="box",
                   cut=0, ax=ax, alpha=0.65, legend=False)
    ax.axvline(THRESHOLD, color="black", ls="--", lw=1.4,
               label=f"Threshold={THRESHOLD}")
    ax.axvspan(-0.05, THRESHOLD, color=C_REAL, alpha=0.04)
    ax.axvspan(THRESHOLD, 1.05,  color=C_FAKE, alpha=0.04)
    ax.set_xlim(-0.05, 1.05)
    ax.set(title="Score Distribution per Source", xlabel="Score", ylabel="")
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8)

    # ── 4. Metric heatmap ─────────────────────────────────────────────────
    ax = axes[1, 1]
    metric_keys  = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    heat_data = np.array([[r[k] for k in metric_keys] for r in rows])
    im = ax.imshow(heat_data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metric_keys)))
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(n_src))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n_src):
        for j in range(len(metric_keys)):
            val  = heat_data[i, j]
            tcol = "black" if 0.35 < val < 0.75 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7.5, color=tcol)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Metric Heatmap per Source")

    plt.tight_layout()
    out = PROJECT / "generator_accuracy_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved -> {out}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="AI image detector batch evaluation")
    p.add_argument("--mode", choices=["batch", "generator"], default="batch",
                   help="'batch' = 200-sample diagnostic (default); "
                        "'generator' = per-source accuracy breakdown")
    p.add_argument("--threshold", type=float, default=None,
                   help=f"Override decision threshold (default: {THRESHOLD})")
    p.add_argument("--max-per-source", type=int, default=200,
                   dest="max_per_source",
                   help="Generator mode: max images sampled per source (default: 200). "
                        "Use 1000 for a full evaluation.")
    return p.parse_args()


def main():
    global THRESHOLD
    args = parse_args()
    if args.threshold is not None:
        THRESHOLD = args.threshold
        print(f"Threshold overridden -> {THRESHOLD}")

    model = load_model()

    if args.mode == "generator":
        print("\n[Mode: generator accuracy]")
        samples = gather_all_samples(max_per_source=args.max_per_source)
        results = run_inference(model, samples)
        tqdm.write("")                             # ensure tqdm bar is fully flushed
        rows    = compute_generator_metrics(results)
        print_generator_report(rows)
        plot_generator_accuracy(rows)

    else:
        print("\n[Mode: batch diagnostic]")
        samples = gather_samples()
        results = run_inference(model, samples)
        metrics = compute_metrics(results)
        print_report(results, metrics)
        plot_results(results, metrics)


if __name__ == "__main__":
    main()
