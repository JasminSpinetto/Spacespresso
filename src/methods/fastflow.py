"""FastFlow: 2D normalizing flow on pretrained backbone features for anomaly detection.

Fully unsupervised — trains on normal images only. Models the distribution of normal
patch features; low-likelihood regions at inference are anomalous.

Architecture:
  1. Frozen pretrained backbone (WideResNet50, same as MLP/PatchCore)
  2. Learnable 1×1 projection: feat_dim → proj_channels
  3. Stack of affine coupling blocks with random channel permutations
  4. Anomaly score = per-pixel negative log-likelihood under the learned distribution
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Backbone (shared logic with MLP/PatchCore)
# ---------------------------------------------------------------------------

def _build_backbone(backbone, out_indices, image_size, device):
    model = timm.create_model(backbone, pretrained=True, features_only=True,
                               out_indices=out_indices).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1).to(device)
    h, w = image_size
    with torch.no_grad():
        feats = model((torch.zeros(1, 3, h, w, device=device) - mean) / std)
    grid_h, grid_w = feats[0].shape[-2], feats[0].shape[-1]
    feat_dim = sum(f.shape[1] for f in feats)
    print(f"Backbone: {backbone} | grid={grid_h}×{grid_w} | feat_dim={feat_dim}")
    return model, mean, std, grid_h, grid_w, feat_dim


@torch.no_grad()
def _extract_features(model, mean, std, imgs, patchsize, device):
    """Extract multi-scale features → (B, C, H, W) keeping spatial dims."""
    imgs  = imgs.to(device, non_blocking=True)
    feats = model((imgs - mean) / std)
    target_hw = feats[0].shape[-2:]
    out = []
    for f in feats:
        f = F.avg_pool2d(f, patchsize, stride=1, padding=patchsize // 2)
        if f.shape[-2:] != target_hw:
            f = F.interpolate(f, size=target_hw, mode="bilinear", align_corners=False)
        out.append(f)
    return torch.cat(out, dim=1)  # (B, C, H, W) — spatial dims preserved


def _load_image(path, image_size):
    h, w = image_size
    with Image.open(path) as im:
        im = im.convert("RGB").resize((w, h), resample=Image.BILINEAR)
        return torch.from_numpy(
            np.asarray(im, dtype=np.float32) / 255.0
        ).permute(2, 0, 1)  # (3, H, W)


# ---------------------------------------------------------------------------
# Normalizing flow components
# ---------------------------------------------------------------------------

class _Permutation(nn.Module):
    """Fixed random orthogonal channel permutation (invertible 1×1 conv)."""

    def __init__(self, channels: int):
        super().__init__()
        W = torch.linalg.qr(torch.randn(channels, channels))[0]
        self.register_buffer("W",     W)
        self.register_buffer("W_inv", W.T)  # orthogonal → inverse = transpose
        self.log_det = 0.0  # orthogonal matrix has log|det|=0

    def forward(self, x: "torch.Tensor", reverse: bool = False) -> "torch.Tensor":
        W = self.W_inv if reverse else self.W
        return F.conv2d(x, W.view(W.shape[0], W.shape[1], 1, 1))


class _AffineCoupling(nn.Module):
    """Affine coupling block: transforms one channel half conditioned on the other."""

    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        half = channels // 2
        self.net = nn.Sequential(
            nn.Conv2d(half, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, half * 2, 3, padding=1),  # → (s, t)
        )
        # Init to identity: last conv outputs zeros → s=0, t=0 → y=x
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: "torch.Tensor", log_det: "torch.Tensor",
                reverse: bool = False):
        x1, x2 = x.chunk(2, dim=1)
        s_raw, t = self.net(x1).chunk(2, dim=1)
        s = torch.tanh(s_raw)  # bounded scale: avoids numerical blow-up
        if not reverse:
            y2 = x2 * torch.exp(s) + t
            log_det = log_det + s.flatten(1).sum(1)  # sum over C×H×W → (B,)
            return torch.cat([x1, y2], dim=1), log_det
        else:
            y2 = (x2 - t) * torch.exp(-s)
            return torch.cat([x1, y2], dim=1), log_det


class _NormalizingFlow(nn.Module):
    """Stack of permutation + affine coupling blocks."""

    def __init__(self, channels: int, hidden_channels: int, n_steps: int):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_steps):
            self.blocks.append(_Permutation(channels))
            self.blocks.append(_AffineCoupling(channels, hidden_channels))

    def forward(self, x: "torch.Tensor"):
        """Returns (nll_per_sample, nll_map).

        nll_per_sample: (B,) — used for training loss
        nll_map: (B, H, W) — used as pixel-level anomaly score at inference
        """
        log_det = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            if isinstance(block, _Permutation):
                x = block(x)
            else:
                x, log_det = block(x, log_det)
        z = x
        # Gaussian NLL per sample: 0.5*||z||² - log_det (plus const)
        nll_sample = 0.5 * z.flatten(1).pow(2).sum(1) - log_det
        # Per-pixel score: sum of squared z-scores across channels
        nll_map    = 0.5 * z.pow(2).sum(dim=1)  # (B, H, W)
        return nll_sample, nll_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_maps(raw: dict) -> dict:
    if not raw:
        return {}
    mn = min(float(np.nanmin(v)) for v in raw.values())
    mx = max(float(np.nanmax(v)) for v in raw.values())
    dr = mx - mn
    if dr <= 1e-12:
        return {k: np.zeros_like(v, dtype=np.float16) for k, v in raw.items()}
    return {k: np.clip((v.astype(np.float32) - mn) / dr, 0.0, 1.0).astype(np.float16)
            for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Public Method
# ---------------------------------------------------------------------------

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        if not _DEPS:
            raise RuntimeError("fastflow requires torch, timm, Pillow, scipy.")

        mc = config.get("method", config)
        dc = config.get("data", {})

        self.seed        = int(config.get("seed", 42))
        isz              = dc.get("image_size", mc.get("image_size", 224))
        self.image_size  = (isz, isz) if isinstance(isz, int) else tuple(isz)

        self.backbone    = str(mc.get("backbone", "wide_resnet50_2"))
        self.out_indices = tuple(int(x) for x in mc.get("out_indices", [2, 3]))
        self.patchsize   = int(mc.get("patchsize", 3))
        self.batch_size  = int(mc.get("batch_size", 16))
        self.epochs      = int(mc.get("epochs", 50))
        self.normals_per_epoch = mc.get("normals_per_epoch", None)
        self.lr          = float(mc.get("lr", 1e-3))
        self.weight_decay= float(mc.get("weight_decay", 1e-5))
        self.proj_channels  = int(mc.get("proj_channels", 128))
        self.flow_steps     = int(mc.get("flow_steps", 8))
        self.flow_hidden_ratio = float(mc.get("flow_hidden_ratio", 1.0))
        self.sigma       = float(mc.get("sigma", 4.0))
        self.class_wise  = bool(mc.get("class_wise", True))
        self.log_every   = int(mc.get("log_every", 10))

        dev = mc.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(dev)

        self.backbone_model, self.mean, self.std, self.grid_h, self.grid_w, self.feat_dim = \
            _build_backbone(self.backbone, self.out_indices, self.image_size, self.device)

        self.flows:       dict[str, _NormalizingFlow] = {}
        self.projections: dict[str, nn.Conv2d]        = {}

    # ------------------------------------------------------------------

    def fit(self, train_data, val_data=None):
        all_samples = list(train_data)
        normals = [s for s in all_samples if s.label == 0]

        if self.class_wise:
            grouped: dict[str, list] = {}
            for s in normals:
                grouped.setdefault(s.class_name, []).append(s)
        else:
            grouped = {GLOBAL_KEY: normals}

        for key, samples in sorted(grouped.items()):
            print(f"\nFastFlow fitting {key}: {len(samples)} normal images")
            flow, proj = self._fit_key(key, samples)
            self.flows[key]       = flow
            self.projections[key] = proj
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        if not self.flows:
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
            k = key if key in self.flows else GLOBAL_KEY
            raw.update(self._predict_key(key_samples, self.flows[k], self.projections[k]))
        return _normalize_maps(raw)

    # ------------------------------------------------------------------

    def _fit_key(self, key: str, normals: list):
        hidden_ch = max(32, int(self.proj_channels * self.flow_hidden_ratio))

        # Learnable projection: feat_dim → proj_channels (1×1 conv, even channels for coupling)
        proj_ch = self.proj_channels + (self.proj_channels % 2)  # ensure even
        proj = nn.Conv2d(self.feat_dim, proj_ch, 1, bias=False).to(self.device)
        nn.init.orthogonal_(proj.weight.reshape(proj_ch, -1).T.reshape(proj.weight.shape))

        flow = _NormalizingFlow(proj_ch, hidden_ch, self.flow_steps).to(self.device)

        optimizer = torch.optim.Adam(
            list(proj.parameters()) + list(flow.parameters()),
            lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6)

        rng = np.random.default_rng(self.seed)
        n_per_epoch = self.normals_per_epoch if self.normals_per_epoch is not None else len(normals)

        best_loss, best_flow_sd, best_proj_sd = float("inf"), None, None

        for epoch in range(1, self.epochs + 1):
            flow.train(); proj.train()
            epoch_losses = []

            perm   = rng.permutation(len(normals))[:n_per_epoch]
            picked = [normals[int(i)] for i in perm]

            for i in range(0, len(picked), self.batch_size):
                batch = picked[i : i + self.batch_size]
                imgs  = torch.stack([_load_image(s.image_path, self.image_size)
                                      for s in batch])

                with torch.no_grad():
                    feats = _extract_features(self.backbone_model, self.mean, self.std,
                                              imgs, self.patchsize, self.device)
                # feats: (B, feat_dim, H, W)
                z = proj(feats)                  # (B, proj_ch, H, W)
                nll, _ = flow(z)                 # nll: (B,)
                loss = nll.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(proj.parameters()) + list(flow.parameters()), 1.0)
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            mean_loss = float(np.mean(epoch_losses))

            if epoch % self.log_every == 0 or epoch == 1:
                print(f"  [{key}] epoch {epoch:03d}/{self.epochs} | "
                      f"loss={mean_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

            if mean_loss < best_loss:
                best_loss    = mean_loss
                best_flow_sd = {k: v.clone() for k, v in flow.state_dict().items()}
                best_proj_sd = {k: v.clone() for k, v in proj.state_dict().items()}

        flow.load_state_dict(best_flow_sd); flow.eval()
        proj.load_state_dict(best_proj_sd); proj.eval()
        return flow, proj

    @torch.no_grad()
    def _predict_key(self, samples: list, flow: "_NormalizingFlow",
                     proj: "nn.Conv2d") -> dict[str, np.ndarray]:
        flow.eval(); proj.eval()
        predictions: dict[str, np.ndarray] = {}

        for i in tqdm(range(0, len(samples), self.batch_size),
                      desc="FastFlow inference", leave=False):
            batch = samples[i : i + self.batch_size]
            imgs  = torch.stack([_load_image(s.image_path, self.image_size)
                                  for s in batch])
            feats = _extract_features(self.backbone_model, self.mean, self.std,
                                      imgs, self.patchsize, self.device)
            z         = proj(feats)
            _, nll_map = flow(z)  # (B, H, W)

            # Upsample NLL map to image size
            nll_up = F.interpolate(
                nll_map.unsqueeze(1), size=self.image_size,
                mode="bilinear", align_corners=False
            ).squeeze(1)  # (B, H, W)

            for j, s in enumerate(batch):
                amap = nll_up[j].cpu().numpy().astype(np.float32)
                if self.sigma > 0:
                    amap = gaussian_filter(amap, sigma=self.sigma).astype(np.float32)
                predictions[str(s.image_id)] = amap.astype(np.float16)

        print(f"  FastFlow [{samples[0].class_name if samples else '?'}] done: {len(samples)} images")
        return predictions
