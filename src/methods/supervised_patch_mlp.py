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
    """Load GT mask at patch grid resolution. Returns (grid_h, grid_w) bool array."""
    with Image.open(mask_path) as im:
        im = im.convert("L").resize((grid_w, grid_h), resample=Image.NEAREST)
        return (np.asarray(im) > 0)  # (grid_h, grid_w)


# Rotationally symmetric classes — all 4 rotations are valid augmentations
_ROTATION_4 = {"class_03", "class_05", "class_06", "class_07"}
# Orientation-sensitive classes — only 180° is safe (object stays upright)
_ROTATION_2 = {"class_01", "class_02", "class_04", "class_08"}


def _augmentations_for_class(class_name: str) -> list[tuple[int, bool]]:
    """Return (k_rot90, flip_horizontal) pairs for this class.

    Horizontal flip is added to each rotation — vertical flip arises naturally
    as flip_H(rot180) and is NOT added separately to avoid duplication.
    Symmetric classes get 4 rotations × 2 flips = 8 unique transforms (D4 group).
    Oriented classes get 2 rotations × 2 flips = 4 unique transforms.
    """
    k_values = [0, 1, 2, 3] if class_name in _ROTATION_4 else [0, 2]
    return [(k, flip) for k in k_values for flip in (False, True)]


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
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
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
        self.augment_anomalies        = bool(mc.get("augment_anomalies", True))
        self.max_imbalance_ratio      = int(mc.get("max_imbalance_ratio", 10))
        self.mlp_dropout              = float(mc.get("mlp_dropout", 0.2))
        self.hard_negative_mining     = bool(mc.get("hard_negative_mining", True))
        self.hnm_epochs               = int(mc.get("hnm_epochs", 10))
        self.hnm_ratio                = int(mc.get("hnm_ratio", 3))  # hard negatives per anomalous patch
        self.sigma                = float(mc.get("sigma", 4.0))
        self.class_wise      = bool(mc.get("class_wise", True))
        self.tta_aggregation = str(mc.get("tta_aggregation", "mean"))  # "mean" or "max"

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
        for key, samples in sorted(grouped.items()):
            print(f"\nFitting MLP for {key}: {len(samples)} images "
                  f"({sum(1 for s in samples if s.label == 1)} anomalous)")
            feats, labels, f_mean, f_std = self._build_patch_dataset(samples)
            self.feat_norms[key] = (f_mean, f_std)
            feats = (feats - f_mean) / f_std          # normalize before training

            # First training pass
            mlp = self._train_mlp(feats, labels, key, self.mlp_epochs)

            # Hard negative mining: find normal patches the MLP scores highest
            if self.hard_negative_mining:
                feats, labels = self._mine_hard_negatives(mlp, feats, labels)
                print(f"  HNM: retrain with {int(labels.sum()):,} anomalous + "
                      f"{int((labels == 0).sum()):,} normal ({self.hnm_epochs} epochs)")
                mlp = self._train_mlp(feats, labels, key, self.hnm_epochs,
                                      init_state=mlp.state_dict())

            self.mlps[key] = mlp
        return self

    def predict(self, test_data, tta: bool = False) -> dict[str, np.ndarray]:
        if not self.mlps:
            raise RuntimeError("Call fit() before predict()")
        samples = list(test_data)
        grouped = self._group(samples) if self.class_wise else {GLOBAL_KEY: samples}
        raw: dict[str, np.ndarray] = {}
        for key, key_samples in grouped.items():
            mlp_key = key if key in self.mlps else GLOBAL_KEY
            if tta:
                print(f"TTA inference [{key}] ({len(key_samples)} images)...")
                raw.update(self._predict_class_tta(key_samples, self.mlps[mlp_key], self.feat_norms[mlp_key]))
            else:
                raw.update(self._predict_class(key_samples, self.mlps[mlp_key], self.feat_norms[mlp_key]))
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
        generator  = torch.Generator()
        generator.manual_seed(self.seed)

        for batch in tqdm(loader, desc="Extracting patch features", leave=False):
            images    = batch["image"]
            patch_feats = _extract_features(self.model, self.mean, self.std,
                                            images, self.patchsize, self.device)
            # patch_feats: (B, H, W, C) → (B, H*W, C)
            B, H, W, C = patch_feats.shape
            flat_feats = patch_feats.reshape(B, H * W, C).cpu()

            for i in range(B):
                label     = int(batch["label"][i].item())
                mask_path = batch["mask_path"][i]
                img_i     = batch["image"][i]  # (3, H_img, W_img)

                if label == 1 and mask_path:
                    mask_grid  = _load_mask(mask_path, H, W)  # (H, W) bool
                    class_name = batch["class_name"][i]
                    augments   = _augmentations_for_class(class_name) if self.augment_anomalies else [(0, False)]
                    for k, do_flip in augments:
                        img_aug  = torch.rot90(img_i, k, dims=[1, 2])
                        mask_aug = np.rot90(mask_grid, k)
                        if do_flip:
                            img_aug  = torch.flip(img_aug, dims=[2])   # horizontal flip
                            mask_aug = np.fliplr(mask_aug)

                        aug_feats = _extract_features(
                            self.model, self.mean, self.std,
                            img_aug.unsqueeze(0), self.patchsize, self.device
                        )  # (1, H, W, C)
                        all_feats.append(aug_feats.reshape(H * W, C).cpu())
                        all_labels.append(torch.from_numpy(
                            mask_aug.reshape(-1).astype(np.float32)
                        ))
                else:
                    # Normal image: randomly subsample to limit memory
                    n_keep = min(self.normal_patches_per_image, H * W)
                    idx = torch.randperm(H * W, generator=generator)[:n_keep]
                    all_feats.append(flat_feats[i][idx])
                    all_labels.append(torch.zeros(n_keep, dtype=torch.float32))

        feats  = torch.cat(all_feats,  dim=0)  # (N_patches, C)
        labels = torch.cat(all_labels, dim=0)  # (N_patches,)
        n_pos  = int(labels.sum().item())
        n_neg  = len(labels) - n_pos

        # Oversample anomalous patches to reach target imbalance ratio
        max_ratio = int(getattr(self, "max_imbalance_ratio", 10))
        if n_pos > 0 and n_neg > n_pos * max_ratio:
            # Keep all anomalous patches; subsample normal to max_ratio * n_pos
            n_neg_keep = n_pos * max_ratio
            neg_idx = torch.where(labels == 0)[0]
            keep    = neg_idx[torch.randperm(len(neg_idx), generator=generator)[:n_neg_keep]]
            pos_idx = torch.where(labels == 1)[0]
            all_idx = torch.cat([pos_idx, keep])
            feats   = feats[all_idx]
            labels  = labels[all_idx]
            n_neg   = n_neg_keep

        n_pos = int(labels.sum().item())
        print(f"  Patch dataset: {len(labels):,} patches | {n_pos:,} anomalous ({100*n_pos/len(labels):.2f}%) | ratio={n_neg//max(n_pos,1)}:1")

        # Compute normalization stats from normal patches only
        normal_feats = feats[labels == 0]
        f_mean = normal_feats.mean(dim=0)
        f_std  = normal_feats.std(dim=0).clamp(min=1e-6)
        return feats, labels, f_mean, f_std

    @torch.no_grad()
    def _mine_hard_negatives(self, mlp, feats, labels):
        """Score all normal patches, keep the hardest ones alongside all anomalous patches."""
        mlp.eval()
        neg_idx = torch.where(labels == 0)[0]
        neg_feats = feats[neg_idx].to(self.device)

        scores = torch.cat([
            torch.sigmoid(mlp(neg_feats[i : i + 4096]))
            for i in range(0, len(neg_feats), 4096)
        ]).cpu()

        n_hard = int(labels.sum().item()) * self.hnm_ratio
        _, hard_rel = torch.topk(scores, min(n_hard, len(scores)))
        hard_idx    = neg_idx[hard_rel]

        # New dataset: all anomalous + all hard negatives
        pos_idx  = torch.where(labels == 1)[0]
        keep     = torch.cat([pos_idx, hard_idx])
        return feats[keep], labels[keep]

    def _train_mlp(self, feats, labels, key, epochs, init_state=None):
        mlp = _MLP(self.feat_dim, self.mlp_hidden, self.mlp_dropout).to(self.device)
        if init_state is not None:
            mlp.load_state_dict(init_state)  # warm-start from previous pass
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.mlp_lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)

        # Compute alpha from class ratio if not set manually
        n_pos   = float(labels.sum().item())
        n_total = float(len(labels))
        alpha   = float(self.focal_alpha) if self.focal_alpha is not None else min(0.95, 1.0 - n_pos / n_total)
        gamma   = self.focal_gamma
        print(f"  Focal loss: alpha={alpha:.3f}, gamma={gamma}")

        dataset = _PatchDataset(feats, labels)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=4096, shuffle=True, num_workers=0)

        best_loss  = float("inf")
        best_state = None

        for epoch in range(1, epochs + 1):
            mlp.train()
            epoch_loss = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = _focal_loss(mlp(xb), yb, alpha=alpha, gamma=gamma)
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())
            scheduler.step()
            mean_loss = float(np.mean(epoch_loss))
            print(f"  [{key}] MLP epoch {epoch:02d}/{epochs} | loss={mean_loss:.5f} | lr={scheduler.get_last_lr()[0]:.2e}")
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

    @torch.no_grad()
    def _predict_class_tta(self, samples, mlp, feat_norm) -> dict[str, np.ndarray]:
        """Predict with test-time augmentation: average over D4/Klein-4 transforms."""
        predictions = {}
        f_mean, f_std = feat_norm
        f_mean = f_mean.to(self.device)
        f_std  = f_std.to(self.device)
        mlp.eval()

        for s in tqdm(samples, desc="MLP inference (TTA)", leave=False):
            img        = _load_image(s.image_path, self.image_size)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
            augments   = _augmentations_for_class(s.class_name)

            # Build batch of all augmented versions
            aug_tensors = []
            for k, do_flip in augments:
                aug = torch.rot90(img_tensor, k, dims=[1, 2])
                if do_flip:
                    aug = torch.flip(aug, dims=[2])
                aug_tensors.append(aug)

            aug_batch   = torch.stack(aug_tensors)  # (n_aug, 3, H, W)
            patch_feats = _extract_features(self.model, self.mean, self.std,
                                            aug_batch, self.patchsize, self.device)
            B, H, W, C  = patch_feats.shape
            flat        = (patch_feats.reshape(B * H * W, C) - f_mean) / f_std
            logits      = mlp(flat).reshape(B, H, W)
            if self.tta_aggregation == "logit":
                # Average in logit space, then apply sigmoid once
                scores_up = F.interpolate(
                    logits.unsqueeze(1), size=self.image_size,
                    mode="bilinear", align_corners=False).squeeze(1)
            else:
                scores_up = F.interpolate(
                    torch.sigmoid(logits).unsqueeze(1), size=self.image_size,
                    mode="bilinear", align_corners=False).squeeze(1)

            # Reverse each transform and collect
            aug_preds = []
            for idx, (k, do_flip) in enumerate(augments):
                pred = scores_up[idx].cpu().numpy().astype(np.float32)
                if do_flip:
                    pred = np.fliplr(pred)
                pred = np.ascontiguousarray(np.rot90(pred, -k % 4))
                aug_preds.append(pred)

            if self.tta_aggregation == "max":
                amap = np.max(aug_preds, axis=0).astype(np.float32)
            elif self.tta_aggregation == "logit":
                amap = torch.sigmoid(torch.tensor(np.mean(aug_preds, axis=0))).numpy().astype(np.float32)
            else:  # mean
                amap = np.mean(aug_preds, axis=0).astype(np.float32)
            if self.sigma > 0:
                amap = gaussian_filter(amap, sigma=self.sigma).astype(np.float32)
            predictions[str(s.image_id)] = amap.astype(np.float16)

        cls = samples[0].class_name if samples else "?"
        n_aug = len(_augmentations_for_class(cls))
        print(f"  TTA [{cls}] done: {len(samples)} images × {n_aug} augments")
        return predictions

    @staticmethod
    def _group(samples) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for s in samples:
            grouped.setdefault(s.class_name, []).append(s)
        return grouped
