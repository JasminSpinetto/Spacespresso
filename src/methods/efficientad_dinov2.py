from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common.augmentation import (
    apply_image_augmentation,
    augmented_sample_count,
    deterministic_seed,
    normalize_augmentation_config,
)
from src.methods.base import BaseMethod


torch = None
nn = None
F = None
Image = None
gaussian_filter = None
torchvision_models = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GLOBAL_BANK_KEY = "__global__"
_TEACHER_CACHE: dict[tuple[str, bool, str], Any] = {}
_DINO_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _require_dependencies() -> None:
    global torch, nn, F, Image, gaussian_filter, torchvision_models
    if (
        torch is not None
        and nn is not None
        and F is not None
        and Image is not None
        and torchvision_models is not None
    ):
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        import torchvision.models as _torchvision_models
        from PIL import Image as _Image
        from scipy.ndimage import gaussian_filter as _gaussian_filter
    except Exception as exc:
        raise RuntimeError(
            "efficientad_dinov2 requires torch, torchvision, Pillow, and scipy. "
            "Install requirements.txt before using this method."
        ) from exc

    torch = _torch
    nn = _nn
    F = _F
    Image = _Image
    gaussian_filter = _gaussian_filter
    torchvision_models = _torchvision_models


def _normalize_image_size(image_size: int | list[int] | tuple[int, int]) -> tuple[int, int]:
    if isinstance(image_size, int):
        return (image_size, image_size)
    if len(image_size) != 2:
        raise ValueError("image_size must be an int or a pair of ints")
    return (int(image_size[0]), int(image_size[1]))


def _load_image(path: str | Path, image_size: tuple[int, int]) -> np.ndarray:
    _require_dependencies()
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _sample_image_array(sample, image_size: tuple[int, int]) -> np.ndarray:
    _require_dependencies()
    image = sample.image if sample.image is not None else _load_image(sample.image_path, image_size)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image for {sample.image_id}, got shape {image.shape}")
    if image.shape[:2] != image_size:
        image = np.asarray(
            Image.fromarray((np.clip(image, 0.0, 1.0) * 255.0).round().astype(np.uint8)).resize(
                (image_size[1], image_size[0]),
                resample=Image.BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0
    return np.clip(image, 0.0, 1.0)


def _normalize_images(images):
    mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (images - mean) / std


class _SampleDataset:
    def __init__(
        self,
        samples,
        image_size: tuple[int, int],
        augmentation_config: dict[str, Any] | None = None,
        seed: int = 42,
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.augmentation_config = normalize_augmentation_config(augmentation_config)
        self.seed = int(seed)
        self.augmentation_enabled = bool(self.augmentation_config["enabled"])
        if self.augmentation_enabled:
            self.variants_per_sample = int(self.augmentation_config["copies_per_image"]) + int(
                bool(self.augmentation_config["include_original"])
            )
        else:
            self.variants_per_sample = 1
        self.variants_per_sample = max(1, self.variants_per_sample)

    def __len__(self) -> int:
        return len(self.samples) * self.variants_per_sample

    def __getitem__(self, index: int):
        sample_index = index // self.variants_per_sample
        variant_index = index % self.variants_per_sample
        sample = self.samples[sample_index]
        image = _sample_image_array(sample, self.image_size)

        if self._should_augment(variant_index):
            image = apply_image_augmentation(
                image,
                self.augmentation_config,
                seed=deterministic_seed(self.seed, sample.image_id, variant_index),
            )

        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "image_id": sample.image_id,
            "class_name": sample.class_name,
        }

    def _should_augment(self, variant_index: int) -> bool:
        if not self.augmentation_enabled:
            return False
        return not (self.augmentation_config["include_original"] and variant_index == 0)


def _build_torchvision_teacher(backbone: str, pretrained: bool, device):
    _require_dependencies()
    name = str(backbone).lower()
    cache_key = (name, bool(pretrained), str(device))
    if cache_key in _TEACHER_CACHE:
        return _TEACHER_CACHE[cache_key]

    if name == "efficientnet_b0":
        weights = (
            torchvision_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        model = torchvision_models.efficientnet_b0(weights=weights).features
    elif name == "mobilenet_v3_small":
        weights = (
            torchvision_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        model = torchvision_models.mobilenet_v3_small(weights=weights).features
    elif name == "mobilenet_v3_large":
        weights = (
            torchvision_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        model = torchvision_models.mobilenet_v3_large(weights=weights).features
    else:
        raise ValueError(
            "Unsupported EfficientAD teacher_backbone. "
            "Use efficientnet_b0, mobilenet_v3_small, or mobilenet_v3_large."
        )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _TEACHER_CACHE[cache_key] = model
    return model


def _build_student(out_channels: int, base_channels: int):
    _require_dependencies()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, channels: int, stride: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class EfficientADStudent(nn.Module):
        def __init__(self, output_channels: int, channels: int):
            super().__init__()
            c = int(channels)
            self.net = nn.Sequential(
                ConvBlock(3, c, stride=2),
                ConvBlock(c, c * 2, stride=2),
                ConvBlock(c * 2, c * 4, stride=2),
                ConvBlock(c * 4, c * 4, stride=2),
                ConvBlock(c * 4, c * 4, stride=2),
                nn.Conv2d(c * 4, int(output_channels), kernel_size=1),
            )

        def forward(self, x):
            return self.net(x)

    return EfficientADStudent(out_channels, base_channels)


def _knn_search(query, bank, bank_chunk_size: int):
    query = query.float()
    bank = bank.float()
    best_dist = torch.full((query.shape[0],), float("inf"), dtype=torch.float32)
    for start in range(0, bank.shape[0], int(bank_chunk_size)):
        bank_chunk = bank[start : start + int(bank_chunk_size)]
        distances = torch.cdist(query, bank_chunk)
        values, _ = distances.min(dim=1)
        best_dist = torch.minimum(best_dist, values.cpu())
    return best_dist


def _random_project(points, out_dim: int | None, seed: int):
    if out_dim is None or points.shape[1] <= int(out_dim):
        return points
    generator = torch.Generator(device=points.device)
    generator.manual_seed(int(seed))
    projection = torch.randn(
        points.shape[1],
        int(out_dim),
        generator=generator,
        device=points.device,
        dtype=points.dtype,
    )
    projection = F.normalize(projection, dim=0)
    return points @ projection


def _greedy_coreset(points, k: int, projection_dim: int | None, seed: int):
    points = points.float()
    n_points = int(points.shape[0])
    if n_points <= int(k):
        return torch.arange(n_points, dtype=torch.long)
    work = _random_project(points, projection_dim, seed)
    generator = torch.Generator(device=work.device)
    generator.manual_seed(int(seed))
    first_idx = int(torch.randint(0, n_points, (1,), generator=generator, device=work.device).item())
    selected = [first_idx]
    min_dist = torch.cdist(work, work[first_idx : first_idx + 1]).squeeze(1)
    for _ in tqdm(range(1, int(k)), desc="DINOv2 coreset", leave=False):
        next_idx = int(torch.argmax(min_dist).item())
        selected.append(next_idx)
        dist_to_new = torch.cdist(work, work[next_idx : next_idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_to_new)
    return torch.tensor(selected, dtype=torch.long)


def _normalize_prediction_dict(
    predictions: dict[str, np.ndarray],
    upper_percentile: float = 99.5,
) -> dict[str, np.ndarray]:
    if not predictions:
        return {}
    lows = []
    highs = []
    for arr in predictions.values():
        x = np.asarray(arr, dtype=np.float32)
        lows.append(float(np.percentile(x, 1.0)))
        highs.append(float(np.percentile(x, upper_percentile)))
    lo = min(lows)
    hi = max(highs)
    if hi - lo <= 1e-8:
        return {image_id: np.zeros_like(arr, dtype=np.float32) for image_id, arr in predictions.items()}
    return {
        image_id: np.clip((np.asarray(arr, dtype=np.float32) - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
        for image_id, arr in predictions.items()
    }


class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        _require_dependencies()

        method_config = config.get("method", config)
        data_config = config.get("data", {})
        self.seed = int(config.get("seed", 42))
        self.image_size = _normalize_image_size(data_config.get("image_size", 224))
        self.batch_size = int(method_config.get("batch_size", 4))
        self.num_workers = int(method_config.get("num_workers", 0))
        self.epochs = int(method_config.get("epochs", 3))
        self.steps_per_epoch = method_config.get("steps_per_epoch")
        self.steps_per_epoch = int(self.steps_per_epoch) if self.steps_per_epoch is not None else None
        self.learning_rate = float(method_config.get("learning_rate", 1e-4))
        self.weight_decay = float(method_config.get("weight_decay", 1e-5))
        self.student_base_channels = int(method_config.get("student_base_channels", 64))
        self.teacher_out_channels = int(method_config.get("teacher_out_channels", 384))
        self.teacher_backbone = str(method_config.get("teacher_backbone", "efficientnet_b0"))
        self.teacher_pretrained = bool(method_config.get("teacher_pretrained", True))
        self.teacher_stats_sample_size = method_config.get("teacher_stats_sample_size", 2000)
        self.teacher_stats_sample_size = (
            None if self.teacher_stats_sample_size is None else int(self.teacher_stats_sample_size)
        )
        self.efficientad_sigma = float(method_config.get("efficientad_sigma", 1.0))
        self.dinov2_sigma = float(method_config.get("dinov2_sigma", 2.0))
        self.efficientad_weight = float(method_config.get("efficientad_weight", 0.75))
        self.dinov2_weight = float(method_config.get("dinov2_weight", 0.25))
        self.normalization_upper_percentile = float(method_config.get("normalization_upper_percentile", 99.5))
        self.augmentation_config = normalize_augmentation_config(
            config.get("augmentation", method_config.get("augmentation", {}))
        )

        self.dinov2_enabled = bool(method_config.get("dinov2_enabled", True))
        self.dinov2_model_name = str(method_config.get("dinov2_model", "dinov2_vits14"))
        self.dinov2_class_wise = bool(method_config.get("dinov2_class_wise", True))
        self.dinov2_candidate_pool_size = int(method_config.get("dinov2_candidate_pool_size", 6000))
        self.dinov2_bank_size = int(method_config.get("dinov2_bank_size", 2000))
        self.dinov2_bank_chunk_size = int(method_config.get("dinov2_bank_chunk_size", 2048))
        self.dinov2_coreset = bool(method_config.get("dinov2_coreset", True))
        self.dinov2_projection_dim = method_config.get("dinov2_projection_dim", 128)
        self.dinov2_projection_dim = (
            None if self.dinov2_projection_dim is None else int(self.dinov2_projection_dim)
        )

        requested_device = method_config.get("device")
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        self.teacher = _build_torchvision_teacher(
            self.teacher_backbone,
            self.teacher_pretrained,
            self.device,
        )
        self.teacher_projector = self._build_teacher_projector().to(self.device)
        self.student = _build_student(self.teacher_out_channels, self.student_base_channels).to(self.device)
        self.dino_model = None

        self.teacher_mean = None
        self.teacher_std = None
        self.training_history: list[dict[str, Any]] = []
        self.dino_banks: dict[str, Any] = {}
        self.dino_grid_shape: tuple[int, int] | None = None

    def fit(self, train_data, val_data=None):
        clean_samples = [sample for sample in train_data if sample.label in (None, 0)]
        if not clean_samples:
            raise ValueError("efficientad_dinov2 requires at least one clean training image")

        self._fit_efficientad(clean_samples)
        if self.dinov2_enabled:
            self._fit_dinov2(clean_samples)
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        samples = list(test_data)
        if self.teacher_mean is None or self.teacher_std is None:
            raise RuntimeError("EfficientAD branch has not been fitted or loaded")

        efficientad_raw = self._predict_efficientad(samples)
        efficientad_maps = _normalize_prediction_dict(
            efficientad_raw,
            upper_percentile=self.normalization_upper_percentile,
        )
        if self.dinov2_enabled:
            if not self.dino_banks:
                raise RuntimeError("DINOv2 branch has not been fitted or loaded")
            dinov2_raw = self._predict_dinov2(samples)
            dinov2_maps = _normalize_prediction_dict(
                dinov2_raw,
                upper_percentile=self.normalization_upper_percentile,
            )
        else:
            dinov2_maps = {image_id: np.zeros_like(pred) for image_id, pred in efficientad_maps.items()}

        total_weight = max(self.efficientad_weight + self.dinov2_weight, 1e-8)
        efficientad_weight = self.efficientad_weight / total_weight
        dinov2_weight = self.dinov2_weight / total_weight
        predictions: dict[str, np.ndarray] = {}
        for image_id in efficientad_maps:
            fused = efficientad_weight * efficientad_maps[image_id] + dinov2_weight * dinov2_maps[image_id]
            predictions[image_id] = np.clip(fused, 0.0, 1.0).astype(np.float32)
        return predictions

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "efficientad_dinov2.pt"
        torch.save(
            {
                "config": self.config,
                "student_state_dict": self.student.state_dict(),
                "teacher_projector_state_dict": self.teacher_projector.state_dict(),
                "teacher_mean": None if self.teacher_mean is None else self.teacher_mean.cpu(),
                "teacher_std": None if self.teacher_std is None else self.teacher_std.cpu(),
                "training_history": self.training_history,
                "dino_banks": self.dino_banks,
                "dino_grid_shape": self.dino_grid_shape,
            },
            path,
        )
        return path

    def load(self, checkpoint_path: str | Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.student.load_state_dict(checkpoint["student_state_dict"])
        self.teacher_projector.load_state_dict(checkpoint["teacher_projector_state_dict"])
        self.teacher_mean = checkpoint["teacher_mean"].to(self.device)
        self.teacher_std = checkpoint["teacher_std"].to(self.device)
        self.training_history = list(checkpoint.get("training_history", []))
        self.dino_banks = {
            str(key): value.cpu().float()
            for key, value in checkpoint.get("dino_banks", {}).items()
        }
        grid_shape = checkpoint.get("dino_grid_shape")
        self.dino_grid_shape = None if grid_shape is None else (int(grid_shape[0]), int(grid_shape[1]))
        return self

    def _build_teacher_projector(self):
        self.teacher.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size[0], self.image_size[1], device=self.device)
            features = self.teacher(_normalize_images(dummy))
        in_channels = int(features.shape[1])
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.seed)
        projector = nn.Conv2d(in_channels, self.teacher_out_channels, kernel_size=1, bias=False)
        with torch.no_grad():
            weight = torch.randn(
                self.teacher_out_channels,
                in_channels,
                1,
                1,
                generator=generator,
                device=self.device,
            )
            weight = weight / max(float(in_channels) ** 0.5, 1.0)
            projector.weight.copy_(weight.cpu())
        for parameter in projector.parameters():
            parameter.requires_grad_(False)
        return projector.eval()

    def _make_loader(self, samples, shuffle: bool = False, augment: bool = False):
        augmentation_config = self.augmentation_config if augment else None
        dataset = _SampleDataset(
            samples,
            self.image_size,
            augmentation_config=augmentation_config,
            seed=self.seed,
        )
        if augment and self.augmentation_config["enabled"]:
            print(
                "EfficientAD/DINOv2 augmentation: "
                f"{len(samples):,} images -> "
                f"{augmented_sample_count(len(samples), self.augmentation_config):,} variants"
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def _teacher_features(self, images, normalize: bool = True):
        with torch.no_grad():
            features = self.teacher(_normalize_images(images))
            features = self.teacher_projector(features)
            if normalize:
                features = (features - self.teacher_mean) / self.teacher_std.clamp_min(1e-6)
        return features

    def _fit_efficientad(self, clean_samples):
        self.teacher_mean, self.teacher_std = self._compute_teacher_stats(clean_samples)
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.student.train()
        self.training_history = []

        for epoch in range(1, self.epochs + 1):
            losses: list[float] = []
            loader = self._make_loader(clean_samples, shuffle=True, augment=True)
            for step, batch in enumerate(tqdm(loader, desc=f"EfficientAD-S epoch {epoch}/{self.epochs}"), start=1):
                if self.steps_per_epoch is not None and step > self.steps_per_epoch:
                    break
                images = batch["image"].to(self.device, non_blocking=True)
                targets = self._teacher_features(images, normalize=True)
                predictions = self.student(_normalize_images(images))
                if predictions.shape[-2:] != targets.shape[-2:]:
                    predictions = F.interpolate(
                        predictions,
                        size=targets.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss = F.mse_loss(predictions, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            self.training_history.append({"epoch": epoch, "loss": mean_loss})
            print(f"EfficientAD-S epoch {epoch}/{self.epochs}: loss={mean_loss:.4f}")

    def _compute_teacher_stats(self, clean_samples):
        samples = list(clean_samples)
        if self.teacher_stats_sample_size is not None and len(samples) > self.teacher_stats_sample_size:
            rng = np.random.default_rng(self.seed)
            indices = rng.choice(len(samples), size=self.teacher_stats_sample_size, replace=False)
            samples = [samples[int(index)] for index in indices]

        loader = self._make_loader(samples, shuffle=False, augment=False)
        channel_sum = None
        channel_sq_sum = None
        count = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="EfficientAD teacher stats"):
                images = batch["image"].to(self.device, non_blocking=True)
                features = self.teacher_projector(self.teacher(_normalize_images(images)))
                dims = (0, 2, 3)
                batch_sum = features.sum(dim=dims)
                batch_sq_sum = (features * features).sum(dim=dims)
                n = int(features.shape[0] * features.shape[2] * features.shape[3])
                channel_sum = batch_sum if channel_sum is None else channel_sum + batch_sum
                channel_sq_sum = batch_sq_sum if channel_sq_sum is None else channel_sq_sum + batch_sq_sum
                count += n
        if count <= 0:
            raise ValueError("No teacher features extracted for EfficientAD stats")
        mean = channel_sum / float(count)
        var = (channel_sq_sum / float(count)) - mean * mean
        std = torch.sqrt(var.clamp_min(1e-8))
        return mean.view(1, -1, 1, 1), std.view(1, -1, 1, 1)

    def _predict_efficientad(self, samples) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        loader = self._make_loader(samples, shuffle=False, augment=False)
        self.student.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="EfficientAD-S inference"):
                images = batch["image"].to(self.device, non_blocking=True)
                targets = self._teacher_features(images, normalize=True)
                outputs = self.student(_normalize_images(images))
                if outputs.shape[-2:] != targets.shape[-2:]:
                    outputs = F.interpolate(outputs, size=targets.shape[-2:], mode="bilinear", align_corners=False)
                maps = torch.mean((outputs - targets) ** 2, dim=1, keepdim=True)
                maps = F.interpolate(maps, size=self.image_size, mode="bilinear", align_corners=False)
                for i, image_id in enumerate(batch["image_id"]):
                    anomaly_map = maps[i, 0].detach().cpu().numpy().astype(np.float32)
                    if self.efficientad_sigma > 0.0:
                        anomaly_map = gaussian_filter(anomaly_map, sigma=self.efficientad_sigma).astype(np.float32)
                    predictions[str(image_id)] = anomaly_map
        return predictions

    def _ensure_dino_model(self):
        if self.dino_model is not None:
            return self.dino_model
        cache_key = (self.dinov2_model_name, str(self.device))
        if cache_key in _DINO_MODEL_CACHE:
            self.dino_model = _DINO_MODEL_CACHE[cache_key]
            return self.dino_model
        try:
            model = torch.hub.load("facebookresearch/dinov2", self.dinov2_model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 via torch.hub. Real runs need network access or a populated torch hub cache."
            ) from exc
        model = model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        _DINO_MODEL_CACHE[cache_key] = model
        self.dino_model = model
        return model

    def _extract_dino_tokens(self, images):
        model = self._ensure_dino_model()
        inputs = _normalize_images(images.to(self.device, non_blocking=True))
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                features = model.forward_features(inputs)
                if isinstance(features, dict):
                    if "x_norm_patchtokens" in features:
                        tokens = features["x_norm_patchtokens"]
                    elif "x_prenorm" in features:
                        tokens = features["x_prenorm"][:, 1:]
                    else:
                        raise RuntimeError(f"Unsupported DINOv2 feature keys: {sorted(features)}")
                else:
                    tokens = features
            elif hasattr(model, "get_intermediate_layers"):
                tokens = model.get_intermediate_layers(inputs, n=1)[0]
            else:
                raise RuntimeError("DINOv2 model does not expose patch-token features")
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected DINOv2 patch tokens with shape B,N,C, got {tuple(tokens.shape)}")
        tokens = F.normalize(tokens.float(), dim=-1)
        n_patches = int(tokens.shape[1])
        grid = int(round(n_patches ** 0.5))
        if grid * grid != n_patches:
            raise RuntimeError(f"DINOv2 patch count is not square: {n_patches}")
        self.dino_grid_shape = (grid, grid)
        return tokens

    def _fit_dinov2(self, clean_samples):
        grouped = self._group_samples(clean_samples) if self.dinov2_class_wise else {GLOBAL_BANK_KEY: clean_samples}
        self.dino_banks = {}
        for class_name, samples in grouped.items():
            print(f"Fitting DINOv2 memory bank for {class_name}: {len(samples)} images")
            self.dino_banks[class_name] = self._fit_dino_bank(samples)

    def _fit_dino_bank(self, samples):
        candidate_bank = None
        candidate_keys = None
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)
        with torch.no_grad():
            for batch in tqdm(self._make_loader(samples, augment=True), desc="DINOv2 feature extraction"):
                tokens = self._extract_dino_tokens(batch["image"])
                flat_tokens = tokens.reshape(-1, tokens.shape[-1]).detach().cpu().float()
                candidate_bank, candidate_keys = self._update_candidate_pool(
                    candidate_bank,
                    candidate_keys,
                    flat_tokens,
                    generator,
                )
        if candidate_bank is None:
            raise ValueError("No DINOv2 patch tokens were extracted")
        n_bank = min(int(self.dinov2_bank_size), int(candidate_bank.shape[0]))
        if self.dinov2_coreset and n_bank < int(candidate_bank.shape[0]):
            selected = _greedy_coreset(
                candidate_bank,
                k=n_bank,
                projection_dim=self.dinov2_projection_dim,
                seed=self.seed,
            )
            bank = candidate_bank[selected].contiguous().cpu()
        else:
            bank = candidate_bank[:n_bank].contiguous().cpu()
        print(f"DINOv2 bank: candidate={candidate_bank.shape[0]:,}, selected={bank.shape[0]:,}")
        return bank

    def _update_candidate_pool(self, candidate_bank, candidate_keys, batch_embeddings, generator):
        pool_size = max(1, int(self.dinov2_candidate_pool_size))
        batch_keys = torch.rand(batch_embeddings.shape[0], generator=generator)
        if candidate_bank is None:
            if batch_embeddings.shape[0] <= pool_size:
                return batch_embeddings.contiguous(), batch_keys.contiguous()
            keep = torch.topk(batch_keys, k=pool_size, largest=False).indices
            return batch_embeddings[keep].contiguous(), batch_keys[keep].contiguous()

        combined_bank = torch.cat([candidate_bank, batch_embeddings], dim=0)
        combined_keys = torch.cat([candidate_keys, batch_keys], dim=0)
        if combined_bank.shape[0] <= pool_size:
            return combined_bank.contiguous(), combined_keys.contiguous()
        keep = torch.topk(combined_keys, k=pool_size, largest=False).indices
        return combined_bank[keep].contiguous(), combined_keys[keep].contiguous()

    def _predict_dinov2(self, samples) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        grouped = self._group_samples(samples) if self.dinov2_class_wise else {GLOBAL_BANK_KEY: samples}
        with torch.no_grad():
            for class_name, class_samples in grouped.items():
                bank_key = class_name if self.dinov2_class_wise else GLOBAL_BANK_KEY
                if bank_key not in self.dino_banks:
                    raise RuntimeError(f"No DINOv2 memory bank found for class '{class_name}'")
                bank = self.dino_banks[bank_key].cpu().float()
                for batch in tqdm(self._make_loader(class_samples), desc=f"DINOv2 inference {class_name}"):
                    tokens = self._extract_dino_tokens(batch["image"])
                    batch_size, n_patches, channels = tokens.shape
                    flat_tokens = tokens.reshape(-1, channels).detach().cpu().float()
                    nn_distances = _knn_search(flat_tokens, bank, self.dinov2_bank_chunk_size)
                    grid_h, grid_w = self.dino_grid_shape
                    patch_scores = nn_distances.view(batch_size, grid_h, grid_w)
                    maps = F.interpolate(
                        patch_scores.unsqueeze(1),
                        size=self.image_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    for i, image_id in enumerate(batch["image_id"]):
                        anomaly_map = maps[i, 0].detach().cpu().numpy().astype(np.float32)
                        if self.dinov2_sigma > 0.0:
                            anomaly_map = gaussian_filter(anomaly_map, sigma=self.dinov2_sigma).astype(np.float32)
                        predictions[str(image_id)] = anomaly_map
        return predictions

    @staticmethod
    def _group_samples(samples) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for sample in samples:
            grouped.setdefault(sample.class_name, []).append(sample)
        return grouped
