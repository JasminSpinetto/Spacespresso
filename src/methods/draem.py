"""DRAEM: Discriminatively-trained Reconstruction Anomaly Embedding Model.

Semi-supervised variant: the reconstructive sub-network is trained on normal images
augmented with synthetic random-blob anomalies; the discriminative sub-network is
additionally fine-tuned on labeled training anomalies with GT masks.

Architecture:
  - Reconstructor: U-Net (3→3) trained to output a clean image given an anomaly-augmented input
  - Discriminator: U-Net (6→1) trained to output an anomaly map from (input, reconstruction)
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
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    _DEPS = True
except ImportError:
    _DEPS = False

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(it, **kw): return it


GLOBAL_KEY = "__global__"

# Rotationally symmetric classes: D4 group (4 rotations × 2 flips = 8 augments)
_ROTATION_4 = {"class_03", "class_05", "class_06", "class_07"}
# Oriented classes: Klein-4 group (0°/180° × 2 flips = 4 augments)
_ROTATION_2 = {"class_01", "class_02", "class_04", "class_08"}


def _augmentations_for_class(class_name: str) -> list[tuple[int, bool]]:
    k_values = [0, 1, 2, 3] if class_name in _ROTATION_4 else [0, 2]
    return [(k, flip) for k in k_values for flip in (False, True)]


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def _load_image(path, image_size):
    h, w = image_size
    with Image.open(path) as im:
        im = im.convert("RGB").resize((w, h), resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _load_mask(path, image_size):
    h, w = image_size
    with Image.open(path) as im:
        im = im.convert("L").resize((w, h), resample=Image.NEAREST)
        return (np.asarray(im) > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Synthetic anomaly generation (no external dataset needed)
# ---------------------------------------------------------------------------

def _smooth_blob_mask(H, W, rng):
    """Smooth irregular blob mask via Gaussian-blurred random field."""
    from scipy.ndimage import gaussian_filter as _gf, zoom as _zm
    h, w = max(H // 4, 8), max(W // 4, 8)
    noise = rng.random((h, w)).astype(np.float32)
    noise = _gf(noise, sigma=rng.uniform(1.5, 4.0))
    noise = _zm(noise, (H / h, W / w), order=1)[:H, :W]
    target = rng.uniform(0.03, 0.35)
    thr    = np.percentile(noise, (1.0 - target) * 100)
    return (noise > thr).astype(np.float32)


def _smooth_texture(H, W, rng):
    """Random smooth color texture."""
    from scipy.ndimage import gaussian_filter as _gf
    tex   = rng.random((H, W, 3)).astype(np.float32)
    sigma = rng.uniform(2.0, 8.0)
    for c in range(3):
        tex[:, :, c] = _gf(tex[:, :, c], sigma=sigma)
    mn, mx = tex.min(), tex.max()
    if mx > mn:
        tex = (tex - mn) / (mx - mn)
    return tex


def _synthetic_anomaly(image_np, rng):
    """Overlay a random smooth texture on a random blob region.

    Returns (augmented float32, mask float32).
    """
    H, W  = image_np.shape[:2]
    mask  = _smooth_blob_mask(H, W, rng)
    tex   = _smooth_texture(H, W, rng)
    alpha = mask[:, :, None]
    aug   = (image_np * (1.0 - alpha) + tex * alpha).astype(np.float32)
    return aug, mask


# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------

class _DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _UNet(nn.Module):
    """Lightweight 4-level U-Net. Image size must be divisible by 16."""

    def __init__(self, in_ch: int, out_ch: int, base: int, sigmoid_out: bool = True):
        super().__init__()
        b = base
        self.pool = nn.MaxPool2d(2)
        self.e1 = _DoubleConv(in_ch, b);    self.e2 = _DoubleConv(b,    b * 2)
        self.e3 = _DoubleConv(b * 2, b * 4); self.e4 = _DoubleConv(b * 4, b * 8)
        self.bot = _DoubleConv(b * 8, b * 16)
        self.u4 = nn.ConvTranspose2d(b * 16, b * 8, 2, 2); self.d4 = _DoubleConv(b * 16, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8,  b * 4, 2, 2); self.d3 = _DoubleConv(b * 8,  b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4,  b * 2, 2, 2); self.d2 = _DoubleConv(b * 4,  b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2,  b,     2, 2); self.d1 = _DoubleConv(b * 2,  b)
        self.out = nn.Conv2d(b, out_ch, 1)
        self.sigmoid_out = sigmoid_out

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b  = self.bot(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(b),  e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        out = self.out(d1)
        return torch.sigmoid(out) if self.sigmoid_out else out


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def _focal_loss(logits, targets, alpha, gamma):
    bce     = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt      = torch.exp(-bce)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt) ** gamma * bce).mean()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_maps(raw):
    if not raw:
        return {}
    mn = min(float(np.nanmin(v)) for v in raw.values())
    mx = max(float(np.nanmax(v)) for v in raw.values())
    dr = mx - mn
    if dr <= 1e-12:
        return {k: np.zeros_like(v, dtype=np.float16) for k, v in raw.items()}
    return {
        k: np.clip((v.astype(np.float32) - mn) / dr, 0.0, 1.0).astype(np.float16)
        for k, v in raw.items()
    }


def _to_tensor(image_np):
    return torch.from_numpy(image_np).permute(2, 0, 1).float()


# ---------------------------------------------------------------------------
# Public Method
# ---------------------------------------------------------------------------

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if not _DEPS:
            raise RuntimeError("draem requires torch, Pillow, scipy.")

        mc = config.get("method", config)
        dc = config.get("data", {})

        self.seed       = int(config.get("seed", 42))
        isz             = dc.get("image_size", mc.get("image_size", 224))
        self.image_size = (isz, isz) if isinstance(isz, int) else tuple(isz)

        if self.image_size[0] % 16 != 0 or self.image_size[1] % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {self.image_size}. "
                             f"Use 224 (=14×16) or 256 (=16×16).")

        self.batch_size        = int(mc.get("batch_size", 8))
        self.epochs            = int(mc.get("epochs", 100))
        self.normals_per_epoch = mc.get("normals_per_epoch", None)  # None = use all
        self.lr                = float(mc.get("lr", 1e-4))
        self.weight_decay      = float(mc.get("weight_decay", 1e-5))
        self.unet_base         = int(mc.get("unet_base", 32))
        self.disc_base         = int(mc.get("disc_base", 16))
        self.lambda_recon      = float(mc.get("lambda_recon", 1.0))
        self.lambda_disc       = float(mc.get("lambda_disc", 2.0))
        self.focal_gamma       = float(mc.get("focal_gamma", 2.0))
        self.focal_alpha       = mc.get("focal_alpha", None)
        self.sigma             = float(mc.get("sigma", 4.0))
        self.class_wise        = bool(mc.get("class_wise", True))
        self.log_every         = int(mc.get("log_every", 10))

        dev = mc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        self.reconstructors: dict[str, _UNet] = {}
        self.discriminators: dict[str, _UNet] = {}

    # ------------------------------------------------------------------

    def fit(self, train_data, val_data=None):
        all_samples = list(train_data)
        if self.class_wise:
            grouped: dict[str, list] = {}
            for s in all_samples:
                grouped.setdefault(s.class_name, []).append(s)
        else:
            grouped = {GLOBAL_KEY: all_samples}

        for key, samples in grouped.items():
            normals   = [s for s in samples if s.label == 0]
            anomalies = [s for s in samples if s.label == 1 and s.mask_path]
            print(f"\nDRAEM fitting {key}: {len(normals)} normal, "
                  f"{len(anomalies)} anomalous (with masks → augmented later)")
            rec, disc = self._fit_key(key, normals, anomalies)
            self.reconstructors[key] = rec
            self.discriminators[key] = disc
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        if not self.reconstructors:
            raise RuntimeError("Call fit() before predict()")

        samples = list(test_data)
        if self.class_wise:
            grouped: dict[str, list] = {}
            for s in samples:
                grouped.setdefault(s.class_name, []).append(s)
        else:
            grouped = {GLOBAL_KEY: samples}

        raw: dict[str, np.ndarray] = {}
        for key, key_samples in grouped.items():
            k = key if key in self.reconstructors else GLOBAL_KEY
            raw.update(self._predict_key(key_samples, self.reconstructors[k],
                                         self.discriminators[k]))
        return _normalize_maps(raw)

    # ------------------------------------------------------------------

    def _fit_key(self, key, normals, anomalies):
        rec  = _UNet(3, 3, self.unet_base, sigmoid_out=True).to(self.device)
        disc = _UNet(6, 1, self.disc_base, sigmoid_out=False).to(self.device)

        # Separate optimizers: reconstruction and discriminator trained independently
        rec_opt  = torch.optim.Adam(rec.parameters(),  lr=self.lr, weight_decay=self.weight_decay)
        disc_opt = torch.optim.Adam(disc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rec_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(rec_opt,  T_max=self.epochs, eta_min=1e-7)
        disc_sched = torch.optim.lr_scheduler.CosineAnnealingLR(disc_opt, T_max=self.epochs, eta_min=1e-7)

        rng = np.random.default_rng(self.seed)

        # Preload anomaly images + D4/Klein-4 augmentations (fits in CPU RAM)
        anom_data: list[tuple] = []
        for s in anomalies:
            img  = _to_tensor(_load_image(s.image_path, self.image_size))
            mask = torch.from_numpy(_load_mask(s.mask_path, self.image_size)).unsqueeze(0)
            for k, do_flip in _augmentations_for_class(s.class_name):
                img_aug  = torch.rot90(img,  k, dims=[1, 2])
                mask_aug = torch.rot90(mask, k, dims=[1, 2])
                if do_flip:
                    img_aug  = torch.flip(img_aug,  dims=[2])
                    mask_aug = torch.flip(mask_aug, dims=[2])
                anom_data.append((img_aug, mask_aug))

        if anom_data:
            print(f"  Anomaly augmentation: {len(anomalies)} → {len(anom_data)} samples "
                  f"({len(anom_data)//max(len(anomalies),1)}x per image)")

        # Focal alpha: auto-compute from estimated anomaly pixel fraction
        if anom_data and self.focal_alpha is None:
            n_pos = sum(m.sum().item() for _, m in anom_data)
            n_tot = len(anom_data) * self.image_size[0] * self.image_size[1]
            alpha = min(0.95, 1.0 - n_pos / max(n_tot, 1))
        else:
            alpha = float(self.focal_alpha) if self.focal_alpha is not None else 0.5
        gamma = self.focal_gamma
        print(f"  Focal: alpha={alpha:.3f}  gamma={gamma}")

        n_per_epoch = (self.normals_per_epoch if self.normals_per_epoch is not None
                       else len(normals))
        print(f"  {n_per_epoch} normals/epoch  |  epochs={self.epochs}  |  "
              f"unet_base={self.unet_base}  disc_base={self.disc_base}")

        best_loss, best_rec_sd, best_disc_sd = float("inf"), None, None

        for epoch in range(1, self.epochs + 1):
            rec.train(); disc.train()
            epoch_losses = []

            # Sample normals for this epoch
            perm   = rng.permutation(len(normals))[:n_per_epoch]
            picked = [normals[int(i)] for i in perm]

            for i in range(0, len(picked), self.batch_size):
                batch = picked[i : i + self.batch_size]

                # Load images
                imgs_np = [_load_image(s.image_path, self.image_size) for s in batch]
                imgs    = torch.stack([_to_tensor(x) for x in imgs_np]).to(self.device)

                # Synthetic anomaly augmentation
                aug_list, mask_list = [], []
                for img_np in imgs_np:
                    aug, msk = _synthetic_anomaly(img_np, rng)
                    aug_list.append(_to_tensor(aug))
                    mask_list.append(torch.from_numpy(msk).unsqueeze(0))
                aug_imgs  = torch.stack(aug_list).to(self.device)
                syn_masks = torch.stack(mask_list).to(self.device)

                # ── Reconstruction pass ──────────────────────────────────────
                reconstruction = rec(aug_imgs)
                recon_loss = (F.mse_loss(reconstruction, imgs) +
                              0.1 * F.l1_loss(reconstruction, imgs))

                rec_opt.zero_grad(set_to_none=True)
                (self.lambda_recon * recon_loss).backward()
                rec_opt.step()

                # ── Discriminator pass (synthetic anomalies) ─────────────────
                with torch.no_grad():
                    recon_det = rec(aug_imgs)  # detach from rec computation graph
                disc_in   = torch.cat([aug_imgs, recon_det], dim=1)
                disc_loss = _focal_loss(disc(disc_in), syn_masks, alpha=alpha, gamma=gamma)

                disc_opt.zero_grad(set_to_none=True)
                (self.lambda_disc * disc_loss).backward()
                disc_opt.step()

                epoch_losses.append(recon_loss.item() + disc_loss.item())

            # ── Real anomaly pass (GT masks — strongest discriminator signal) ─
            if anom_data:
                for i in range(0, len(anom_data), self.batch_size):
                    batch_a = anom_data[i : i + self.batch_size]
                    anom_imgs  = torch.stack([x[0] for x in batch_a]).to(self.device)
                    anom_masks = torch.stack([x[1] for x in batch_a]).to(self.device)

                    with torch.no_grad():
                        anom_recon = rec(anom_imgs)
                    disc_in_real = torch.cat([anom_imgs, anom_recon], dim=1)
                    real_loss    = _focal_loss(disc(disc_in_real), anom_masks, alpha=alpha, gamma=gamma)

                    disc_opt.zero_grad(set_to_none=True)
                    (self.lambda_disc * real_loss).backward()
                    disc_opt.step()
                    epoch_losses.append(real_loss.item())

            rec_sched.step(); disc_sched.step()

            mean_loss = float(np.mean(epoch_losses))
            if epoch % self.log_every == 0 or epoch == 1:
                print(f"  [{key}] epoch {epoch:03d}/{self.epochs} | "
                      f"loss={mean_loss:.5f} | lr={rec_sched.get_last_lr()[0]:.2e}")
            if mean_loss < best_loss:
                best_loss    = mean_loss
                best_rec_sd  = {k: v.clone() for k, v in rec.state_dict().items()}
                best_disc_sd = {k: v.clone() for k, v in disc.state_dict().items()}

        rec.load_state_dict(best_rec_sd);   rec.eval()
        disc.load_state_dict(best_disc_sd); disc.eval()
        return rec, disc

    @torch.no_grad()
    def _predict_key(self, samples, rec, disc):
        predictions: dict[str, np.ndarray] = {}
        rec.eval(); disc.eval()

        for i in tqdm(range(0, len(samples), self.batch_size),
                      desc="DRAEM inference", leave=False):
            batch = samples[i : i + self.batch_size]
            imgs  = torch.stack([
                _to_tensor(_load_image(s.image_path, self.image_size))
                for s in batch
            ]).to(self.device)

            reconstruction = rec(imgs)
            disc_in        = torch.cat([imgs, reconstruction], dim=1)
            amap           = torch.sigmoid(disc(disc_in))  # (B, 1, H, W)

            for j, s in enumerate(batch):
                pred = amap[j, 0].cpu().numpy().astype(np.float32)
                if self.sigma > 0:
                    pred = gaussian_filter(pred, sigma=self.sigma).astype(np.float32)
                predictions[str(s.image_id)] = pred.astype(np.float16)

        return predictions
