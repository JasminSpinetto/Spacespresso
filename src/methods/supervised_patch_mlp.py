"""Supervised Patch MLP: semi-supervised anomaly detection using labeled training anomalies.

Feature extraction uses a pretrained backbone (same as PatchCore).
For each training image:
  - Good images: all patches labeled normal (0)
  - Anomalous images: patches overlapping the GT mask labeled anomalous (1)
A lightweight MLP binary classifier is trained on these patch features.
At inference, each patch is scored by the MLP and the map is upsampled to image size.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.methods.base import BaseMethod

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import timm
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    _DEPS = True
except ImportError:
    _DEPS = False

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kw): return it

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
GLOBAL_KEY    = "__global__"


def _require():
    if not _DEPS:
        raise RuntimeError("supervised_patch_mlp requires torch, timm, Pillow, scipy.")


def _normalize_image_size(s):
    return (s, s) if isinstance(s, int) else (int(s[0]), int(s[1]))


# ---------------------------------------------------------------------------
# Feature extractor (same as PatchCore)
# ---------------------------------------------------------------------------

def _build_feature_extractor(backbone, out_indices, image_size, device):
    model = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=out_indices).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1).to(device)

    # Probe feature dims and grid shape
    with torch.no_grad():
        h, w = image_size
        dummy = torch.zeros(1, 3, h, w, device=device)
        feats = model((dummy - mean) / std)
    grid_h, grid_w = feats[0].shape[-2], feats[0].shape[-1]
    feat_dim = sum(f.shape[1] for f in feats)
    print(f"Backbone: {backbone} | grid={grid_h}×{grid_w} | feat_dim={feat_dim}")
    return model, mean, std, grid_h, grid_w, feat_dim


@torch.no_grad()
def _extract_features(model, mean, std, images_tensor, patchsize, device):
    """Extract and concat multi-scale patch features → (B, H, W, C)."""
    images_tensor = images_tensor.to(device, non_blocking=True)
    feats = model((images_tensor - mean) / std)
    target_hw = feats[0].shape[-2:]
    processed = []
    for feat in feats:
        feat = F.avg_pool2d(feat, kernel_size=patchsize, stride=1, padding=patchsize // 2)
        if feat.shape[-2:] != target_hw:
            feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        processed.append(feat)
    out = torch.cat(processed, dim=1)           # (B, C, H, W)
    return out.permute(0, 2, 3, 1).contiguous() # (B, H, W, C)


# ---------------------------------------------------------------------------
# GT mask → patch-level labels
# ---------------------------------------------------------------------------

def _load_image(path, image_size):
    with Image.open(path) as im:
        im = im.convert("RGB").resize(image_size[::-1], resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _load_mask(mask_path, grid_h, grid_w):
    """Load GT mask, downsample to patch grid resolution. Returns (grid_h, grid_w) bool array."""
    with Image.open(mask_path) as im:
        im = im.convert("L").resize((grid_w, grid_h), resample=Image.NEAREST)
        return (np.asarray(im) > 0).reshape(-1)  # (grid_h*grid_w,)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def _focal_loss(logits: "torch.Tensor", targets: "torch.Tensor",
                alpha: float, gamma: float) -> "torch.Tensor":
    """Binary focal loss.

    alpha: weight for positive class (set higher when positives are rare).
    gamma: focusing exponent — higher gamma down-weights easy examples more.
    """
    bce      = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt       = torch.exp(-bce)                                    # probability of correct class
    alpha_t  = targets * alpha + (1.0 - targets) * (1.0 - alpha) # per-sample alpha
    return (alpha_t * (1.0 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# MLP classifier
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class _PatchDataset(torch.utils.data.Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features
        self.labels   = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class _SampleDataset:
    def __init__(self, samples, image_size):
        self.samples    = list(samples)
        self.image_size = image_size

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = s.image if s.image is not None else _load_image(s.image_path, self.image_size)
        img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        return {
            "image":      torch.from_numpy(img).permute(2, 0, 1).contiguous(),
            "image_id":   s.image_id,
            "class_name": s.class_name,
            "mask_path":  str(s.mask_path) if s.mask_path else "",
            "label":      int(s.label) if s.label is not None else 0,
        }


# ---------------------------------------------------------------------------
# Normalization (same as PatchCore)
# ---------------------------------------------------------------------------

def _normalize_maps(raw):
    if not raw: return {}
    mn = min(float(np.nanmin(v)) for v in raw.values())
    mx = max(float(np.nanmax(v)) for v in raw.values())
    dr = mx - mn
    if dr <= 1e-12:
        return {k: np.zeros_like(v, dtype=np.float16) for k, v in raw.items()}
    return {
        k: np.clip((v.astype(np.float32) - mn) / dr, 0.0, 1.0).astype(np.float16)
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# Public Method
# ---------------------------------------------------------------------------

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        _require()

        mc = config.get("method", config)
        dc = config.get("data", {})

        self.seed            = int(config.get("seed", 42))
        self.image_size      = _normalize_image_size(dc.get("image_size", 224))
        self.backbone        = str(mc.get("backbone", "wide_resnet50_2"))
        self.out_indices     = tuple(int(x) for x in mc.get("out_indices", (2, 3)))
        self.patchsize       = int(mc.get("patchsize", 3))
        self.batch_size      = int(mc.get("batch_size", 8))
        self.mlp_hidden      = int(mc.get("mlp_hidden", 256))
        self.mlp_epochs      = int(mc.get("mlp_epochs", 20))
        self.mlp_lr          = float(mc.get("mlp_lr", 1e-3))
        self.weight_decay    = float(mc.get("weight_decay", 1e-5))
        self.focal_gamma          = float(mc.get("focal_gamma", 2.0))
        self.focal_alpha          = mc.get("focal_alpha", None)
        self.normal_patches_per_image = int(mc.get("normal_patches_per_image", 50))
        self.sigma                = float(mc.get("sigma", 4.0))
        self.class_wise      = bool(mc.get("class_wise", True))

        device = mc.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.model, self.mean, self.std, self.grid_h, self.grid_w, self.feat_dim = \
            _build_feature_extractor(self.backbone, self.out_indices, self.image_size, self.device)

        self.mlps:       dict[str, _MLP]                          = {}
        self.feat_norms: dict[str, tuple["torch.Tensor", "torch.Tensor"]] = {}  # key → (mean, std)

    # ------------------------------------------------------------------

    def fit(self, train_data, val_data=None):
        all_samples = list(train_data)
        grouped = self._group(all_samples) if self.class_wise else {GLOBAL_KEY: all_samples}
        for key, samples in grouped.items():
            print(f"\nFitting MLP for {key}: {len(samples)} images "
                  f"({sum(1 for s in samples if s.label == 1)} anomalous)")
            feats, labels, f_mean, f_std = self._build_patch_dataset(samples)
            self.feat_norms[key] = (f_mean, f_std)
            feats = (feats - f_mean) / f_std          # normalize before training
            self.mlps[key] = self._train_mlp(feats, labels, key)
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        if not self.mlps:
            raise RuntimeError("Call fit() before predict()")
        samples = list(test_data)
        grouped = self._group(samples) if self.class_wise else {GLOBAL_KEY: samples}
        raw: dict[str, np.ndarray] = {}
        for key, key_samples in grouped.items():
            mlp_key = key if key in self.mlps else GLOBAL_KEY
            raw.update(self._predict_class(key_samples, self.mlps[mlp_key],
                                           self.feat_norms[mlp_key]))
        return _normalize_maps(raw)

    # ------------------------------------------------------------------

    def _build_patch_dataset(self, samples):
        loader = torch.utils.data.DataLoader(
            _SampleDataset(samples, self.image_size),
            batch_size=self.batch_size, shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        all_feats  = []
        all_labels = []

        for batch in tqdm(loader, desc="Extracting patch features", leave=False):
            images    = batch["image"]
            patch_feats = _extract_features(self.model, self.mean, self.std,
                                            images, self.patchsize, self.device)
            # patch_feats: (B, H, W, C) → (B, H*W, C)
            B, H, W, C = patch_feats.shape
            flat_feats = patch_feats.reshape(B, H * W, C).cpu()

            for i in range(B):
                label      = int(batch["label"][i].item())
                mask_path  = batch["mask_path"][i]
                patch_flat = flat_feats[i]  # (H*W, C)

                if label == 1 and mask_path:
                    # Anomalous image: keep ALL patches with their GT labels
                    patch_labels = torch.from_numpy(
                        _load_mask(mask_path, H, W).astype(np.float32)
                    )
                    all_feats.append(patch_flat)
                    all_labels.append(patch_labels)
                else:
                    # Normal image: randomly subsample to limit memory
                    n_keep = min(self.normal_patches_per_image, H * W)
                    idx = torch.randperm(H * W)[:n_keep]
                    all_feats.append(patch_flat[idx])
                    all_labels.append(torch.zeros(n_keep, dtype=torch.float32))

        feats  = torch.cat(all_feats,  dim=0)  # (N_patches, C)
        labels = torch.cat(all_labels, dim=0)  # (N_patches,)
        n_pos  = int(labels.sum().item())
        print(f"  Patch dataset: {len(labels):,} patches | {n_pos:,} anomalous ({100*n_pos/len(labels):.2f}%)")

        # Compute normalization stats from normal patches only
        normal_feats = feats[labels == 0]
        f_mean = normal_feats.mean(dim=0)
        f_std  = normal_feats.std(dim=0).clamp(min=1e-6)
        return feats, labels, f_mean, f_std

    def _train_mlp(self, feats, labels, key):
        mlp = _MLP(self.feat_dim, self.mlp_hidden).to(self.device)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.mlp_lr, weight_decay=self.weight_decay)

        # Compute alpha from class ratio if not set manually
        n_pos   = float(labels.sum().item())
        n_total = float(len(labels))
        alpha   = float(self.focal_alpha) if self.focal_alpha is not None else (1.0 - n_pos / n_total)
        gamma   = self.focal_gamma
        print(f"  Focal loss: alpha={alpha:.3f}, gamma={gamma}")

        dataset = _PatchDataset(feats, labels)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=0)

        best_loss  = float("inf")
        best_state = None

        for epoch in range(1, self.mlp_epochs + 1):
            mlp.train()
            epoch_loss = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = _focal_loss(mlp(xb), yb, alpha=alpha, gamma=gamma)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            mean_loss = float(np.mean(epoch_loss))
            print(f"  [{key}] MLP epoch {epoch:02d}/{self.mlp_epochs} | loss={mean_loss:.5f}")
            if mean_loss < best_loss:
                best_loss  = mean_loss
                best_state = {k: v.clone() for k, v in mlp.state_dict().items()}

        mlp.load_state_dict(best_state)
        mlp.eval()
        return mlp

    @torch.no_grad()
    def _predict_class(self, samples, mlp, feat_norm) -> dict[str, np.ndarray]:
        predictions = {}
        f_mean, f_std = feat_norm
        f_mean = f_mean.to(self.device)
        f_std  = f_std.to(self.device)

        loader = torch.utils.data.DataLoader(
            _SampleDataset(samples, self.image_size),
            batch_size=self.batch_size, shuffle=False, num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        mlp.eval()

        for batch in tqdm(loader, desc="MLP inference", leave=False):
            images = batch["image"]
            patch_feats = _extract_features(self.model, self.mean, self.std,
                                            images, self.patchsize, self.device)
            B, H, W, C = patch_feats.shape
            flat = (patch_feats.reshape(B * H * W, C) - f_mean) / f_std  # normalize
            scores = torch.sigmoid(mlp(flat)).reshape(B, H, W)

            # Upsample to image size
            scores_up = F.interpolate(
                scores.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(1)

            for i, image_id in enumerate(batch["image_id"]):
                amap = scores_up[i].cpu().numpy().astype(np.float32)
                if self.sigma > 0:
                    amap = gaussian_filter(amap, sigma=self.sigma).astype(np.float32)
                predictions[str(image_id)] = amap.astype(np.float16)

        return predictions

    @staticmethod
    def _group(samples) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for s in samples:
            grouped.setdefault(s.class_name, []).append(s)
        return grouped
