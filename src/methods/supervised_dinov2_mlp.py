"""Supervised DINOv2 MLP: semi-supervised anomaly detection using DINOv2 patch tokens.
Similar to SupervisedPatchMLP but uses DINOv2 as the backbone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

from src.methods.base import BaseMethod

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
GLOBAL_KEY    = "__global__"

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

def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pt = torch.exp(-bce)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    return (alpha_t * (1.0 - pt) ** gamma * bce).mean()

def _load_mask(mask_path, grid_h, grid_w):
    with Image.open(mask_path) as im:
        im = im.convert("L").resize((grid_w, grid_h), resample=Image.NEAREST)
        return (np.asarray(im) > 0)

# Rotationally symmetric classes
_ROTATION_4 = {"class_03", "class_05", "class_06", "class_07"}

def _augmentations_for_class(class_name: str) -> list[tuple[int, bool]]:
    k_values = [0, 1, 2, 3] if class_name in _ROTATION_4 else [0, 2]
    return [(k, flip) for k in k_values for flip in (False, True)]

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        mc = config.get("method", {})
        dc = config.get("data", {})

        self.device = torch.device(mc.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = mc.get("dinov2_model", "dinov2_vits14")
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name).to(self.device).eval()
        
        self.image_size = dc.get("image_size", 224)
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        
        self.mlp_hidden = int(mc.get("mlp_hidden", 256))
        self.mlp_epochs = int(mc.get("mlp_epochs", 20))
        self.mlp_lr = float(mc.get("mlp_lr", 1e-3))
        self.weight_decay = float(mc.get("weight_decay", 1e-5))
        self.focal_gamma = float(mc.get("focal_gamma", 2.0))
        self.focal_alpha = mc.get("focal_alpha", None)
        self.normal_patches_per_image = int(mc.get("normal_patches_per_image", 50))
        self.max_imbalance_ratio = int(mc.get("max_imbalance_ratio", 10))
        self.mlp_dropout = float(mc.get("mlp_dropout", 0.2))
        self.sigma = float(mc.get("sigma", 4.0))
        self.tta_aggregation = mc.get("tta_aggregation", "mean")
        self.class_wise = bool(mc.get("class_wise", True))
        self.seed = int(config.get("seed", 42))

        self.mlps = {}
        self.feat_norms = {}
        
        # Probe features
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size[0], self.image_size[1]).to(self.device)
            tokens = self._extract_tokens(dummy)
            self.feat_dim = tokens.shape[-1]
            self.grid_h = self.grid_w = int(tokens.shape[1]**0.5)
            print(f"DINOv2 Grid: {self.grid_h}x{self.grid_w} | Dim: {self.feat_dim}")

    def _extract_tokens(self, images):
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(self.device)
        std  = torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1).to(self.device)
        x = (images.to(self.device) - mean) / std
        if hasattr(self.model, "get_intermediate_layers"):
            tokens = self.model.get_intermediate_layers(x, n=1)[0]
        else:
            tokens = self.model.forward_features(x)["x_norm_patchtokens"]
        return F.normalize(tokens.float(), dim=-1)

    def fit(self, train_data, val_data=None):
        all_samples = list(train_data)
        grouped = self._group(all_samples) if self.class_wise else {GLOBAL_KEY: all_samples}
        
        for key, samples in sorted(grouped.items()):
            print(f"\nFitting DINOv2-MLP for {key}: {len(samples)} images")
            feats, labels, f_mean, f_std = self._build_patch_dataset(samples)
            self.feat_norms[key] = (f_mean, f_std)
            feats = (feats - f_mean) / f_std
            
            # Pass 1: Initial training
            print(f"  Pass 1: Training initial MLP ({self.mlp_epochs} epochs)...")
            mlp = self._train_mlp(feats, labels, self.mlp_epochs)
            
            # Pass 2: Hard Negative Mining
            print("  Pass 2: Mining hard negatives and retraining...")
            mlp.eval()
            with torch.no_grad():
                neg_idx = torch.where(labels == 0)[0]
                neg_feats = feats[neg_idx].to(self.device)
                scores = torch.cat([torch.sigmoid(mlp(neg_feats[i:i+4096])).cpu() 
                                  for i in range(0, len(neg_feats), 4096)])
                
                n_hard = int(labels.sum().item()) * 3
                _, hard_rel = torch.topk(scores, min(n_hard, len(scores)))
                hard_idx = neg_idx[hard_rel]
                pos_idx  = torch.where(labels == 1)[0]
                keep     = torch.cat([pos_idx, hard_idx])
                
            mlp = self._train_mlp(feats[keep], labels[keep], self.mlp_epochs // 2, init_state=mlp.state_dict())
            self.mlps[key] = mlp
        return self

    def _train_mlp(self, feats, labels, epochs, init_state=None):
        mlp = _MLP(self.feat_dim, self.mlp_hidden, self.mlp_dropout).to(self.device)
        if init_state: mlp.load_state_dict(init_state)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=self.mlp_lr, weight_decay=self.weight_decay)
        n_pos = float(labels.sum().item())
        alpha = float(self.focal_alpha) if self.focal_alpha is not None else min(0.95, 1.0 - n_pos / len(labels))
        
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(feats, labels), 
                                           batch_size=4096, shuffle=True)
        for epoch in range(1, epochs + 1):
            mlp.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                loss = _focal_loss(mlp(xb), yb, alpha=alpha, gamma=self.focal_gamma)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            if epoch % 5 == 0 or epoch == 1:
                print(f"    Epoch {epoch:02d}/{epochs} | loss={np.mean(losses):.5f}")
        return mlp.eval()

    def _build_patch_dataset(self, samples):
        all_feats = []
        all_labels = []
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        for s in tqdm(samples, desc="Extracting DINOv2 tokens", leave=False):
            img = s.image if s.image is not None else self._load_image(s.image_path)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                tokens = self._extract_tokens(img_tensor) # (1, N, C)
                B, N, C = tokens.shape
                tokens = tokens.view(N, C).cpu()
                
            if s.label == 1 and s.mask_path:
                mask = _load_mask(s.mask_path, self.grid_h, self.grid_w).flatten()
                all_feats.append(tokens)
                all_labels.append(torch.from_numpy(mask.astype(np.float32)))
            else:
                n_keep = min(self.normal_patches_per_image, N)
                idx = torch.randperm(N, generator=generator)[:n_keep]
                all_feats.append(tokens[idx])
                all_labels.append(torch.zeros(n_keep, dtype=torch.float32))

        feats = torch.cat(all_feats, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        normal_feats = feats[labels == 0]
        f_mean = normal_feats.mean(dim=0)
        f_std = normal_feats.std(dim=0).clamp(min=1e-6)
        
        return feats, labels, f_mean, f_std

    def _load_image(self, path):
        with Image.open(path) as im:
            im = im.convert("RGB").resize(self.image_size[::-1], resample=Image.BILINEAR)
            return np.asarray(im, dtype=np.float32) / 255.0

    def predict(self, test_data, tta: bool = False) -> dict[str, np.ndarray]:
        predictions = {}
        samples = list(test_data)
        grouped = self._group(samples) if self.class_wise else {GLOBAL_KEY: samples}
        
        for key, key_samples in grouped.items():
            mlp_key = key if key in self.mlps else GLOBAL_KEY
            mlp = self.mlps[mlp_key]
            f_mean, f_std = self.feat_norms[mlp_key]
            f_mean, f_std = f_mean.to(self.device), f_std.to(self.device)
            
            for s in tqdm(key_samples, desc=f"Inference {key}", leave=False):
                img = self._load_image(s.image_path)
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                
                if tta:
                    augments = _augmentations_for_class(s.class_name)
                    aug_tensors = []
                    for k, do_flip in augments:
                        aug = torch.rot90(img_tensor, k, dims=[1, 2])
                        if do_flip: aug = torch.flip(aug, dims=[2])
                        aug_tensors.append(aug)
                    
                    batch = torch.stack(aug_tensors).to(self.device)
                    with torch.no_grad():
                        tokens = self._extract_tokens(batch) # (8, N, C)
                        B, N, C = tokens.shape
                        flat = (tokens.view(B*N, C) - f_mean) / f_std
                        scores = torch.sigmoid(mlp(flat)).view(B, N)
                        
                    # Reverse transforms and average
                    grid_scores = scores.view(B, self.grid_h, self.grid_w)
                    aug_preds = []
                    for i, (k, do_flip) in enumerate(augments):
                        pred = grid_scores[i].cpu().numpy()
                        if do_flip: pred = np.fliplr(pred)
                        pred = np.rot90(pred, -k % 4)
                        aug_preds.append(pred)
                    
                    final_grid = np.mean(aug_preds, axis=0)
                else:
                    with torch.no_grad():
                        tokens = self._extract_tokens(img_tensor.unsqueeze(0).to(self.device))
                        flat = (tokens.squeeze(0) - f_mean) / f_std
                        scores = torch.sigmoid(mlp(flat)).view(self.grid_h, self.grid_w)
                        final_grid = scores.cpu().numpy()
                
                # Upsample
                amap = Image.fromarray(final_grid).resize(self.image_size[::-1], resample=Image.BILINEAR)
                amap = np.asarray(amap, dtype=np.float32)
                if self.sigma > 0:
                    amap = gaussian_filter(amap, sigma=self.sigma)
                predictions[str(s.image_id)] = amap.astype(np.float16)
                
        return predictions

    def _group(self, samples) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for s in samples:
            grouped.setdefault(s.class_name, []).append(s)
        return grouped
