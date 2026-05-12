"""Uninformed Students: Student-Teacher anomaly detection.

Based on Bergmann et al., CVPR 2020.
Teacher: frozen pretrained ResNet18 producing dense local descriptors.
Students: ensemble of patch CNNs trained to regress the teacher on normal images.
Anomaly score: regression error + predictive variance, normalized on training data.
"""
from __future__ import annotations

import copy
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
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
GLOBAL_KEY    = "__global__"


def _require_dependencies() -> None:
    if not _DEPS_AVAILABLE:
        raise RuntimeError(
            "student_teacher requires torch, timm, Pillow, and scipy. "
            "Install requirements.txt before using this method."
        )


def _normalize_image_size(image_size):
    if isinstance(image_size, int):
        return (image_size, image_size)
    return (int(image_size[0]), int(image_size[1]))


# ---------------------------------------------------------------------------
# Loss / stats helpers (ported from lab notebook)
# ---------------------------------------------------------------------------

def _rgb_to_gray(x):
    weights = x.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _augment_patches(patches, gray_prob=0.1, noise_std=0.05):
    patches = patches.clone()
    flip_h = torch.rand(patches.shape[0], device=patches.device) < 0.5
    if flip_h.any():
        patches[flip_h] = torch.flip(patches[flip_h], dims=[-1])
    flip_v = torch.rand(patches.shape[0], device=patches.device) < 0.5
    if flip_v.any():
        patches[flip_v] = torch.flip(patches[flip_v], dims=[-2])
    to_gray = torch.rand(patches.shape[0], device=patches.device) < gray_prob
    if to_gray.any():
        patches[to_gray] = _rgb_to_gray(patches[to_gray]).repeat(1, 3, 1, 1)
    if noise_std > 0:
        patches = (patches + noise_std * torch.randn_like(patches)).clamp(0.0, 1.0)
    return patches


def _student_loss(student_out, teacher_target):
    return ((student_out - teacher_target) ** 2).sum(dim=-1).mean()


def _regression_error_map(students_pred, teacher_pred):
    mean_students = students_pred.mean(dim=1)
    return ((mean_students - teacher_pred) ** 2).sum(dim=-1)


def _variance_map(students_pred):
    mean_sq = (students_pred ** 2).sum(dim=-1).mean(dim=1)
    mean_s  = students_pred.mean(dim=1)
    return mean_sq - (mean_s ** 2).sum(dim=-1)


def _update_vector_stats(sum_v, sum_sq_v, count, x):
    x = x.reshape(-1, x.shape[-1]).float()
    if sum_v is None:
        sum_v    = x.sum(dim=0)
        sum_sq_v = (x ** 2).sum(dim=0)
    else:
        sum_v    = sum_v    + x.sum(dim=0)
        sum_sq_v = sum_sq_v + (x ** 2).sum(dim=0)
    return sum_v, sum_sq_v, count + x.shape[0]


def _finalize_vector_stats(sum_v, sum_sq_v, count, eps=1e-6):
    mean = sum_v / max(count, 1)
    var  = ((sum_sq_v / max(count, 1)) - mean.pow(2)).clamp_min(eps)
    return mean, var, torch.sqrt(var)


def _update_scalar_stats(sum_s, sum_sq_s, count, x):
    x = x.reshape(-1).float()
    if sum_s is None:
        sum_s    = x.sum()
        sum_sq_s = (x ** 2).sum()
    else:
        sum_s    = sum_s    + x.sum()
        sum_sq_s = sum_sq_s + (x ** 2).sum()
    return sum_s, sum_sq_s, count + x.numel()


def _finalize_scalar_stats(sum_s, sum_sq_s, count, eps=1e-6):
    mean = sum_s / max(count, 1)
    var  = ((sum_sq_s / max(count, 1)) - mean.pow(2)).clamp_min(eps)
    return mean, var, torch.sqrt(var)


# ---------------------------------------------------------------------------
# Multi-pooling helper modules (required for patch sizes 33 and 65)
# ---------------------------------------------------------------------------

class _MultiPoolPrepare(nn.Module):
    def __init__(self, patch_y, patch_x):
        super().__init__()
        self.pad_top    = int(np.ceil( (patch_y - 1) / 2))
        self.pad_bottom = int(np.floor((patch_y - 1) / 2))
        self.pad_left   = int(np.ceil( (patch_x - 1) / 2))
        self.pad_right  = int(np.floor((patch_x - 1) / 2))

    def forward(self, x):
        return F.pad(x, [self.pad_left, self.pad_right, self.pad_top, self.pad_bottom])


class _UnwrapPrepare(nn.Module):
    def forward(self, x):
        x = F.pad(x, [0, -1, 0, -1])
        return x.contiguous().view(x.shape[0], -1).transpose(0, 1).contiguous()


class _UnwrapPool(nn.Module):
    def __init__(self, out_chans, cur_h, cur_w, d_h, d_w):
        super().__init__()
        self.out_chans = int(out_chans)
        self.cur_h = int(cur_h)
        self.cur_w = int(cur_w)
        self.d_h   = int(d_h)
        self.d_w   = int(d_w)

    def forward(self, x):
        y = x.view(self.out_chans, self.cur_w, self.cur_h, self.d_h, self.d_w, -1)
        return y.transpose(2, 3).contiguous()


class _MultiMaxPooling(nn.Module):
    def __init__(self, k_w, k_h, d_w, d_h):
        super().__init__()
        self.paddings = [(-j, -i) for i in range(d_h) for j in range(d_w)]
        self.layers   = nn.ModuleList([
            nn.MaxPool2d(kernel_size=(k_w, k_h), stride=(d_w, d_h))
            for _ in self.paddings
        ])

    def forward(self, x):
        outputs = []
        for layer, (pl, pt) in zip(self.layers, self.paddings):
            y = F.pad(x, [pl, pl, pt, pt])
            outputs.append(layer(y))
        max_h = max(o.shape[2] for o in outputs)
        max_w = max(o.shape[3] for o in outputs)
        padded = []
        for y in outputs:
            h, w = y.shape[2], y.shape[3]
            pt = int(np.floor((max_h - h) / 2));  pb = int(np.ceil((max_h - h) / 2))
            pl = int(np.floor((max_w - w) / 2));  pr = int(np.ceil((max_w - w) / 2))
            padded.append(F.pad(y, [pl, pr, pt, pb]))
        return torch.cat(padded, dim=0)


# ---------------------------------------------------------------------------
# Teacher: frozen pretrained ResNet18
# ---------------------------------------------------------------------------

class _Teacher(nn.Module):
    output_dim = 512

    def __init__(self, patch_size: int, patch_batch_size: int = 1024, stride: int = 1):
        super().__init__()
        self.patch_size       = int(patch_size)
        self.patch_batch_size = int(patch_batch_size)
        self.stride           = int(stride)
        self.model = timm.create_model(
            "resnet18", pretrained=True, num_classes=0, global_pool="avg"
        ).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std",  torch.tensor(IMAGENET_STD ).view(1, 3, 1, 1), persistent=False)

    @torch.no_grad()
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

    @torch.no_grad()
    def dense_latent(self, x):
        b, _, h, w = x.shape
        pad     = self.patch_size // 2
        x_pad   = F.pad(x, [pad, pad, pad, pad])
        patches = F.unfold(x_pad, kernel_size=self.patch_size, stride=self.stride)
        # output positions per side
        oh = (h + 2 * pad - self.patch_size) // self.stride + 1
        ow = (w + 2 * pad - self.patch_size) // self.stride + 1
        patches = patches.transpose(1, 2).reshape(-1, 3, self.patch_size, self.patch_size)
        feats   = torch.cat([
            self.forward(patches[s : s + self.patch_batch_size])
            for s in range(0, patches.shape[0], self.patch_batch_size)
        ], dim=0)
        return feats.view(b, oh, ow, self.output_dim)


# ---------------------------------------------------------------------------
# Student: patch CNN (supports patch sizes 17, 33, 65)
# ---------------------------------------------------------------------------

class _Student(nn.Module):
    def __init__(self, patch_size: int, output_dim: int = 128):
        super().__init__()
        self.patch_size  = int(patch_size)
        self.output_dim  = int(output_dim)
        self.latent_dim  = self.output_dim
        self.act         = nn.LeakyReLU(5e-3)
        self.max_pool    = nn.MaxPool2d(2, 2)
        self.multi_pool_prepare = _MultiPoolPrepare(patch_size, patch_size)
        self.multi_max_pooling  = _MultiMaxPooling(2, 2, 2, 2)
        self.unwrap_prepare     = _UnwrapPrepare()

        if patch_size == 17:
            self.conv1 = nn.Conv2d(3,   128, 6, 1)
            self.conv2 = nn.Conv2d(128, 256, 5, 1)
            self.conv3 = nn.Conv2d(256, 256, 5, 1)
            self.conv4 = nn.Conv2d(256, self.output_dim, 4, 1)
            self.out_chans   = self.output_dim
            self.pool_stages = 0
        elif patch_size == 33:
            self.conv1 = nn.Conv2d(3,   128, 5, 1)
            self.conv2 = nn.Conv2d(128, 256, 5, 1)
            self.conv3 = nn.Conv2d(256, 256, 2, 1)
            self.conv4 = nn.Conv2d(256, self.output_dim, 4, 1)
            self.out_chans   = self.output_dim
            self.pool_stages = 2
        elif patch_size == 65:
            self.conv1 = nn.Conv2d(3,   128, 5, 1)
            self.conv2 = nn.Conv2d(128, 128, 5, 1)
            self.conv3 = nn.Conv2d(128, 256, 5, 1)
            self.conv4 = nn.Conv2d(256, 256, 4, 1)
            self.conv5 = nn.Conv2d(256, self.output_dim, 1, 1)
            self.out_chans   = self.output_dim
            self.pool_stages = 3
        else:
            raise ValueError(f"patch_size must be 17, 33, or 65; got {patch_size}")

        self.decoder = nn.Linear(self.latent_dim, 512)

    def dense_latent(self, x):
        h, w = x.shape[-2:]
        divisor = 2 ** self.pool_stages
        if h % divisor != 0 or w % divisor != 0:
            raise ValueError(
                f"image size {(h, w)} must be divisible by {divisor} for patch_size={self.patch_size}"
            )
        x = self.multi_pool_prepare(x)

        if self.patch_size == 17:
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = self.act(self.conv3(x))
            x = self.act(self.conv4(x))

        elif self.patch_size == 33:
            up2 = _UnwrapPool(self.out_chans, h // 4, w // 4, 2, 2)
            up1 = _UnwrapPool(self.out_chans, h // 2, w // 2, 2, 2)
            x = self.act(self.conv1(x));   x = self.multi_max_pooling(x)
            x = self.act(self.conv2(x));   x = self.multi_max_pooling(x)
            x = self.act(self.conv3(x))
            x = self.act(self.conv4(x))
            x = self.unwrap_prepare(x);    x = up2(x);    x = up1(x)

        else:  # 65
            up3 = _UnwrapPool(self.out_chans, h // 8, w // 8, 2, 2)
            up2 = _UnwrapPool(self.out_chans, h // 4, w // 4, 2, 2)
            up1 = _UnwrapPool(self.out_chans, h // 2, w // 2, 2, 2)
            x = self.act(self.conv1(x));   x = self.multi_max_pooling(x)
            x = self.act(self.conv2(x));   x = self.multi_max_pooling(x)
            x = self.act(self.conv3(x));   x = self.multi_max_pooling(x)
            x = self.act(self.conv4(x))
            x = self.act(self.conv5(x))
            x = self.unwrap_prepare(x);    x = up3(x);    x = up2(x);    x = up1(x)

        y = x.view(self.out_chans, h, w, -1)
        return y.permute(3, 1, 2, 0).contiguous()  # (B, H, W, out_chans)


# ---------------------------------------------------------------------------
# Normalization (same pattern as patchcore_lite)
# ---------------------------------------------------------------------------

def _normalize_maps(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

class _SampleDataset:
    def __init__(self, samples, image_size):
        self.samples    = list(samples)
        self.image_size = image_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = s.image if s.image is not None else _load_image(s.image_path, self.image_size)
        img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
        return {
            "image":      torch.from_numpy(img).permute(2, 0, 1).contiguous(),
            "image_id":   s.image_id,
            "class_name": s.class_name,
        }


def _load_image(path, image_size):
    with Image.open(path) as im:
        im = im.convert("RGB").resize(image_size[::-1], resample=Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _compute_teacher_stats(teacher, loader, device):
    teacher.eval()
    sum_v = sum_sq_v = None
    count = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Teacher stats p={teacher.patch_size}", leave=False):
            desc = teacher.dense_latent(batch["image"].to(device, non_blocking=True))
            sum_v, sum_sq_v, count = _update_vector_stats(sum_v, sum_sq_v, count, desc)
    mean, var, std = _finalize_vector_stats(sum_v, sum_sq_v, count)
    return {"mean": mean, "var": var, "std": std}


def _train_students(students, teacher, teacher_stats, loader, device, epochs, lr, wd, student_hw=None):
    teacher.eval()
    mu  = teacher_stats["mean"].view(1, 1, 1, -1)
    std = teacher_stats["std" ].view(1, 1, 1, -1)
    optimizers  = [torch.optim.Adam(s.parameters(), lr=lr, weight_decay=wd) for s in students]
    best_states = [copy.deepcopy(s.state_dict()) for s in students]
    best_losses = [float("inf")] * len(students)
    history     = []

    for epoch in range(1, epochs + 1):
        for s in students:
            s.train()
        epoch_losses = [[] for _ in students]

        for batch in tqdm(loader, desc=f"Students p={teacher.patch_size} epoch {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                target = (teacher.dense_latent(images) - mu) / std
                oh, ow = target.shape[1], target.shape[2]
            imgs_s = F.interpolate(images, size=student_hw, mode="bilinear", align_corners=False) if student_hw else images
            for idx, (s, opt) in enumerate(zip(students, optimizers)):
                opt.zero_grad(set_to_none=True)
                loss = _student_loss(_pool_to(s.dense_latent(imgs_s), oh, ow), target)
                loss.backward()
                opt.step()
                epoch_losses[idx].append(loss.item())

        mean_losses = [float(np.mean(l)) for l in epoch_losses]
        history.append(mean_losses)
        print(f"[ST] p={teacher.patch_size} epoch {epoch:02d} | " +
              " | ".join(f"S{i}: {l:.5f}" for i, l in enumerate(mean_losses)))
        for idx, loss in enumerate(mean_losses):
            if loss < best_losses[idx]:
                best_losses[idx] = loss
                best_states[idx] = copy.deepcopy(students[idx].state_dict())

    for s, state in zip(students, best_states):
        s.load_state_dict(state)
        s.eval()
    return students, history


def _compute_score_stats(teacher, students, teacher_stats, loader, device, student_hw=None):
    teacher.eval()
    for s in students:
        s.eval()
    mu  = teacher_stats["mean"].view(1, 1, 1, -1)
    std = teacher_stats["std" ].view(1, 1, 1, -1)
    err_sum = err_sq = var_sum = var_sq = None
    err_n = var_n = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Score calibration p={teacher.patch_size}", leave=False):
            images = batch["image"].to(device, non_blocking=True)
            t_pred = (teacher.dense_latent(images) - mu) / std
            oh, ow = t_pred.shape[1], t_pred.shape[2]
            imgs_s = F.interpolate(images, size=student_hw, mode="bilinear", align_corners=False) if student_hw else images
            s_pred = torch.stack([_pool_to(s.dense_latent(imgs_s), oh, ow) for s in students], dim=1)
            err_sum, err_sq, err_n = _update_scalar_stats(err_sum, err_sq, err_n, _regression_error_map(s_pred, t_pred))
            var_sum, var_sq, var_n = _update_scalar_stats(var_sum, var_sq, var_n, _variance_map(s_pred))

    err_mean, _, err_std = _finalize_scalar_stats(err_sum, err_sq, err_n)
    var_mean, _, var_std = _finalize_scalar_stats(var_sum, var_sq, var_n)
    return {"err_mean": err_mean, "err_std": err_std, "var_mean": var_mean, "var_std": var_std}


def _pool_to(student_out, oh, ow):
    """Pool student dense output (B, H, W, C) to (B, oh, ow, C)."""
    _, h, w, _ = student_out.shape
    if h == oh and w == ow:
        return student_out
    x = F.adaptive_avg_pool2d(student_out.permute(0, 3, 1, 2), (oh, ow))
    return x.permute(0, 2, 3, 1).contiguous()


def _score_batch(images, scale, student_hw=None):
    teacher  = scale["teacher"]
    students = scale["students"]
    t_stats  = scale["teacher_stats"]
    s_stats  = scale["score_stats"]
    mu  = t_stats["mean"].view(1, 1, 1, -1)
    std = t_stats["std" ].view(1, 1, 1, -1)
    t_pred = (teacher.dense_latent(images) - mu) / std
    oh, ow = t_pred.shape[1], t_pred.shape[2]
    imgs_s = F.interpolate(images, size=student_hw, mode="bilinear", align_corners=False) if student_hw else images
    s_pred = torch.stack([_pool_to(s.dense_latent(imgs_s), oh, ow) for s in students], dim=1)
    err = (_regression_error_map(s_pred, t_pred) - s_stats["err_mean"]) / s_stats["err_std"]
    var = (_variance_map(s_pred)                 - s_stats["var_mean"]) / s_stats["var_std"]
    return err + var  # (B, oh, ow)


# ---------------------------------------------------------------------------
# Public Method class
# ---------------------------------------------------------------------------

class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        _require_dependencies()

        mc = config.get("method", config)
        dc = config.get("data", {})

        self.seed                = int(config.get("seed", 42))
        self.image_size          = _normalize_image_size(dc.get("image_size", 224))
        self.patch_sizes         = tuple(int(p) for p in mc.get("patch_sizes", [17]))
        self.n_students          = int(mc.get("n_students", 3))
        self.student_epochs      = int(mc.get("student_epochs", 20))
        self.student_lr          = float(mc.get("student_lr", 1e-4))
        self.weight_decay        = float(mc.get("weight_decay", 1e-5))
        self.student_batch_size  = int(mc.get("student_batch_size", 1))
        self.teacher_patch_batch = int(mc.get("teacher_patch_batch_size", 1024))
        self.teacher_stride      = int(mc.get("teacher_stride", 4))
        _sisize = mc.get("student_input_size")
        self.student_hw = (int(_sisize), int(_sisize)) if _sisize is not None else None
        self.sigma               = float(mc.get("sigma", 4.0))
        self.class_wise          = bool(mc.get("class_wise", True))
        self.cal_fraction        = float(mc.get("cal_fraction", 0.1))

        device = mc.get("device")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.scales_per_class: dict[str, list] = {}

    def fit(self, train_data, val_data=None):
        clean = [s for s in train_data if s.label in (None, 0)]
        if not clean:
            raise ValueError("Student-Teacher requires at least one clean training image")
        grouped = self._group(clean) if self.class_wise else {GLOBAL_KEY: clean}
        for cls, samples in grouped.items():
            print(f"Training Student-Teacher for {cls}: {len(samples)} images")
            self.scales_per_class[cls] = self._fit_class(samples)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        if not self.scales_per_class:
            raise RuntimeError("Call fit() before predict()")
        samples = list(test_data)
        grouped = self._group(samples) if self.class_wise else {GLOBAL_KEY: samples}
        raw: dict[str, np.ndarray] = {}
        for cls, cls_samples in grouped.items():
            key = cls if self.class_wise else GLOBAL_KEY
            if key not in self.scales_per_class:
                raise RuntimeError(f"No model for class '{cls}'")
            raw.update(self._predict_class(cls_samples, self.scales_per_class[key]))
        return _normalize_maps(raw)

    def _fit_class(self, samples: list) -> list:
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(len(samples))
        n_cal         = max(1, int(len(samples) * self.cal_fraction))
        cal_samples   = [samples[i] for i in idx[:n_cal]]
        train_samples = [samples[i] for i in idx[n_cal:]]

        train_loader = self._make_loader(train_samples, shuffle=True)
        cal_loader   = self._make_loader(cal_samples,   shuffle=False)

        scales = []
        for patch_size in self.patch_sizes:
            teacher  = _Teacher(patch_size, self.teacher_patch_batch, self.teacher_stride).to(self.device).eval()
            students = [_Student(patch_size, teacher.output_dim).to(self.device) for _ in range(self.n_students)]
            print(f"  patch_size={patch_size} | stride={self.teacher_stride} | {self.n_students} students x "
                  f"{sum(p.numel() for p in students[0].parameters()):,} params")

            t_stats  = _compute_teacher_stats(teacher, train_loader, self.device)
            students, history = _train_students(students, teacher, t_stats, train_loader,
                                               self.device, self.student_epochs, self.student_lr, self.weight_decay,
                                               student_hw=self.student_hw)
            s_stats  = _compute_score_stats(teacher, students, t_stats, cal_loader, self.device,
                                            student_hw=self.student_hw)

            scales.append({
                "patch_size":       patch_size,
                "teacher":          teacher,
                "training_history": history,
                "students":      students,
                "teacher_stats": t_stats,
                "score_stats":   s_stats,
            })
        return scales

    @torch.no_grad()
    def _predict_class(self, samples: list, scales: list) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        loader = self._make_loader(samples, shuffle=False)

        for batch in tqdm(loader, desc="Student-Teacher inference"):
            images = batch["image"].to(self.device, non_blocking=True)
            maps   = [_score_batch(images, sc, student_hw=self.student_hw) for sc in scales]
            score  = torch.stack(maps, dim=0).mean(dim=0)  # (B, H, W)

            score_up = F.interpolate(
                score.unsqueeze(1), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(1)

            for i, image_id in enumerate(batch["image_id"]):
                amap = score_up[i].cpu().numpy().astype(np.float32)
                if self.sigma > 0:
                    amap = gaussian_filter(amap, sigma=self.sigma).astype(np.float32)
                predictions[str(image_id)] = amap.astype(np.float16)

        return predictions

    def plot_training_history(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available")
            return
        for cls, scales in self.scales_per_class.items():
            for scale in scales:
                history = scale.get("training_history", [])
                if not history:
                    continue
                epochs = list(range(1, len(history) + 1))
                n_students = len(history[0])
                _, ax = plt.subplots(figsize=(7, 4))
                for s_idx in range(n_students):
                    ax.plot(epochs, [h[s_idx] for h in history], label=f"Student {s_idx}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"{cls} | patch_size={scale['patch_size']}")
                ax.legend()
                plt.tight_layout()
                plt.show()

    @staticmethod
    def _group(samples) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for s in samples:
            grouped.setdefault(s.class_name, []).append(s)
        return grouped

    def _make_loader(self, samples, shuffle: bool = False):
        dataset = _SampleDataset(samples, self.image_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.student_batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
