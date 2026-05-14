from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from src.methods.base import BaseMethod


torch = None
nn = None
F = None
Image = None
ImageDraw = None
ImageFilter = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable


def _require_dependencies() -> None:
    global torch, nn, F, Image, ImageDraw, ImageFilter
    if torch is not None and nn is not None and F is not None and Image is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        from PIL import Image as _Image
        from PIL import ImageDraw as _ImageDraw
        from PIL import ImageFilter as _ImageFilter
    except Exception as exc:
        raise RuntimeError(
            "synthetic_unet requires torch and Pillow. Install requirements.txt before using it."
        ) from exc

    torch = _torch
    nn = _nn
    F = _F
    Image = _Image
    ImageDraw = _ImageDraw
    ImageFilter = _ImageFilter


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


def _stable_seed(seed: int, *parts: object) -> int:
    import hashlib

    payload = "::".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


class _SyntheticAnomalyDataset:
    def __init__(
        self,
        samples,
        image_size: tuple[int, int],
        config: dict[str, Any],
        seed: int,
        length: int | None = None,
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.config = config
        self.seed = int(seed)
        self.length = int(length if length is not None else len(self.samples))
        self.clean_probability = float(config.get("clean_probability", 0.15))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        sample = self.samples[index % len(self.samples)]
        image = _sample_image_array(sample, self.image_size)
        rng = np.random.default_rng(_stable_seed(self.seed, sample.image_id, index))

        if rng.random() < self.clean_probability:
            synthetic = image.copy()
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            synthetic, mask = _make_synthetic_anomaly(image, rng, self.config)

        return {
            "image": torch.from_numpy(synthetic).permute(2, 0, 1).contiguous(),
            "mask": torch.from_numpy(mask[None, ...]).contiguous(),
        }


class _PredictionDataset:
    def __init__(self, samples, image_size: tuple[int, int]):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = _sample_image_array(sample, self.image_size)
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1).contiguous(),
            "image_id": sample.image_id,
        }


def _build_unet(base_channels: int):
    _require_dependencies()

    class ConvBlock(nn.Module):
        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.net(x)

    class SmallUNet(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            c = int(channels)
            self.enc1 = ConvBlock(3, c)
            self.enc2 = ConvBlock(c, c * 2)
            self.enc3 = ConvBlock(c * 2, c * 4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(c * 4, c * 8)
            self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
            self.dec3 = ConvBlock(c * 8, c * 4)
            self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
            self.dec2 = ConvBlock(c * 4, c * 2)
            self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
            self.dec1 = ConvBlock(c * 2, c)
            self.head = nn.Conv2d(c, 1, kernel_size=1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.head(d1)

    return SmallUNet(base_channels)


class Method(BaseMethod):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        _require_dependencies()

        method_config = config.get("method", config)
        data_config = config.get("data", {})
        self.seed = int(config.get("seed", 42))
        self.image_size = _normalize_image_size(data_config.get("image_size", 224))
        self.batch_size = int(method_config.get("batch_size", 8))
        self.epochs = int(method_config.get("epochs", 8))
        self.steps_per_epoch = method_config.get("steps_per_epoch")
        self.steps_per_epoch = int(self.steps_per_epoch) if self.steps_per_epoch is not None else None
        self.learning_rate = float(method_config.get("learning_rate", 1e-3))
        self.weight_decay = float(method_config.get("weight_decay", 1e-4))
        self.base_channels = int(method_config.get("base_channels", 24))
        self.num_workers = int(method_config.get("num_workers", 0))
        self.threshold = float(method_config.get("threshold", 0.5))
        self.synthetic_config = dict(method_config.get("synthetic", {}))
        validation_config = config.get("validation", {})
        self.validation_metric = str(method_config.get("validation_metric", validation_config.get("metric", "pixel_ap")))
        self.validation_interval = int(method_config.get("validation_interval", 1))
        self.keep_best = bool(method_config.get("keep_best", True))
        self.training_history: list[dict[str, Any]] = []
        self.best_epoch: int | None = None
        self.best_metrics: dict[str, Any] = {}

        requested_device = method_config.get("device")
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)
        self.model = _build_unet(base_channels=self.base_channels).to(self.device)

    def fit(self, train_data, val_data=None):
        clean_samples = [sample for sample in train_data if sample.label in (None, 0)]
        if not clean_samples:
            raise ValueError("Synthetic U-Net requires at least one clean training image")

        self.model.train()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        epoch_length = (
            self.steps_per_epoch * self.batch_size
            if self.steps_per_epoch is not None
            else len(clean_samples)
        )
        dataset = _SyntheticAnomalyDataset(
            clean_samples,
            image_size=self.image_size,
            config=self.synthetic_config,
            seed=self.seed,
            length=epoch_length,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        validation_samples = list(val_data or [])
        best_score = -float("inf")
        best_state_dict = None
        self.training_history = []
        self.best_epoch = None
        self.best_metrics = {}

        for epoch in range(1, self.epochs + 1):
            losses: list[float] = []
            for batch in tqdm(loader, desc=f"Synthetic U-Net epoch {epoch}/{self.epochs}"):
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = _segmentation_loss(logits, masks)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            mean_loss = float(np.mean(losses)) if losses else 0.0
            print(f"Synthetic U-Net epoch {epoch}/{self.epochs}: loss={mean_loss:.4f}")

            record: dict[str, Any] = {"epoch": epoch, "loss": mean_loss}
            if validation_samples and self.validation_interval > 0 and (
                epoch % self.validation_interval == 0 or epoch == self.epochs
            ):
                metrics = self._evaluate_validation(validation_samples)
                record.update({f"val_{key}": value for key, value in metrics.items()})
                score = metrics.get(self.validation_metric)
                if score is None:
                    raise ValueError(f"Validation metric {self.validation_metric} is undefined")
                print(
                    f"Synthetic U-Net validation epoch {epoch}: "
                    f"{self.validation_metric}={float(score):.4f}, "
                    f"pixel_ap={metrics['pixel_ap']:.4f}, image_ap={metrics['image_ap']:.4f}"
                )
                if self.keep_best and float(score) > best_score:
                    best_score = float(score)
                    self.best_epoch = epoch
                    self.best_metrics = copy.deepcopy(metrics)
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self.model.state_dict().items()
                    }
                    print(f"Synthetic U-Net best checkpoint updated at epoch {epoch}")
                self.model.train()
            self.training_history.append(record)

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)
            print(
                f"Synthetic U-Net restored best epoch {self.best_epoch} "
                f"({self.validation_metric}={self.best_metrics[self.validation_metric]:.4f})"
            )
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        samples = list(test_data)
        dataset = _PredictionDataset(samples, self.image_size)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        predictions: dict[str, np.ndarray] = {}
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Synthetic U-Net inference"):
                images = batch["image"].to(self.device, non_blocking=True)
                logits = self.model(images)
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                for i, image_id in enumerate(batch["image_id"]):
                    predictions[str(image_id)] = probs[i, 0].astype(np.float32)
        return predictions

    def _evaluate_validation(self, val_samples) -> dict[str, Any]:
        from src.common.evaluation import evaluate_predictions
        from src.common.prediction_processing import process_prediction_maps

        samples = list(val_samples)
        predictions = self.predict(samples)
        predictions = process_prediction_maps(samples, predictions, self.config)
        return evaluate_predictions(samples, predictions).as_dict()

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "synthetic_unet.pt"
        torch.save(
            {
                "config": self.config,
                "state_dict": self.model.state_dict(),
                "training_history": self.training_history,
                "best_epoch": self.best_epoch,
                "best_metrics": self.best_metrics,
            },
            path,
        )
        return path

    def load(self, checkpoint_path: str | Path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.training_history = list(checkpoint.get("training_history", []))
        self.best_epoch = checkpoint.get("best_epoch")
        self.best_metrics = dict(checkpoint.get("best_metrics", {}))
        return self


def _make_synthetic_anomaly(
    image: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    mask = _random_blob_mask((height, width), rng, config)
    if bool(config.get("restrict_to_foreground", True)):
        foreground = image.mean(axis=2) > float(config.get("foreground_threshold", 0.08))
        mask = mask * foreground.astype(np.float32)
        if float(mask.sum()) < 8:
            mask = _random_blob_mask((height, width), rng, {**config, "restrict_to_foreground": False})

    texture = _random_texture(image.shape, rng, config)
    alpha = float(rng.uniform(float(config.get("alpha_min", 0.45)), float(config.get("alpha_max", 0.9))))
    mask3 = mask[..., None]
    synthetic = image * (1.0 - mask3 * alpha) + texture * (mask3 * alpha)
    return np.clip(synthetic, 0.0, 1.0).astype(np.float32), mask.astype(np.float32)


def _random_blob_mask(
    shape: tuple[int, int],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> np.ndarray:
    _require_dependencies()
    height, width = shape
    min_area_fraction = float(config.get("min_area_fraction", 0.002))
    max_area_fraction = float(config.get("max_area_fraction", 0.08))
    n_blobs = int(rng.integers(int(config.get("min_blobs", 1)), int(config.get("max_blobs", 4)) + 1))

    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    area = height * width
    for _ in range(n_blobs):
        blob_area = float(rng.uniform(min_area_fraction, max_area_fraction) * area / n_blobs)
        radius = max(3.0, np.sqrt(blob_area / np.pi))
        rx = int(radius * rng.uniform(0.7, 1.8))
        ry = int(radius * rng.uniform(0.7, 1.8))
        rx = max(1, min(rx, max(1, width // 2 - 1)))
        ry = max(1, min(ry, max(1, height // 2 - 1)))
        cx = int(rng.integers(rx, max(rx + 1, width - rx)))
        cy = int(rng.integers(ry, max(ry + 1, height - ry)))
        if rng.random() < 0.55:
            draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
        else:
            points = []
            for angle in np.linspace(0, 2 * np.pi, int(rng.integers(5, 9)), endpoint=False):
                jitter = rng.uniform(0.55, 1.25)
                points.append((cx + int(np.cos(angle) * rx * jitter), cy + int(np.sin(angle) * ry * jitter)))
            draw.polygon(points, fill=255)

    blur_radius = float(config.get("mask_blur_radius", 1.2))
    if blur_radius > 0:
        mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    mask = np.asarray(mask_img, dtype=np.float32) / 255.0
    return (mask > 0.35).astype(np.float32)


def _random_texture(
    shape: tuple[int, int, int],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> np.ndarray:
    height, width, channels = shape
    lowres = int(config.get("texture_lowres", 24))
    noise = rng.random((lowres, lowres, channels), dtype=np.float32)
    texture_img = Image.fromarray((noise * 255.0).astype(np.uint8), mode="RGB")
    texture_img = texture_img.resize((width, height), resample=Image.BILINEAR)
    texture = np.asarray(texture_img, dtype=np.float32) / 255.0
    color = rng.uniform(0.05, 0.95, size=(1, 1, channels)).astype(np.float32)
    color_weight = float(config.get("color_weight", 0.45))
    texture = texture * (1.0 - color_weight) + color * color_weight
    if rng.random() < float(config.get("invert_probability", 0.25)):
        texture = 1.0 - texture
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def _segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs + targets, dim=dims)
    dice = 1.0 - torch.mean((2.0 * intersection + 1.0) / (denominator + 1.0))
    return bce + dice
