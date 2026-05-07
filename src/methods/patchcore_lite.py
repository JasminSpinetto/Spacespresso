from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.methods.base import BaseMethod


torch = None
nn = None
F = None
timm = None
gaussian_filter = None
Image = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    def tqdm(iterable, **kwargs):
        return iterable


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GLOBAL_BANK_KEY = "__global__"


def _require_dependencies() -> None:
    global torch, nn, F, timm, gaussian_filter, Image
    if torch is not None and nn is not None and F is not None and timm is not None:
        return
    try:
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
        import timm as _timm
        from PIL import Image as _Image
        from scipy.ndimage import gaussian_filter as _gaussian_filter
    except Exception as exc:
        raise RuntimeError(
            "patchcore_lite requires torch, timm, Pillow, and scipy. "
            "Install requirements.txt before using this method."
        ) from exc

    torch = _torch
    nn = _nn
    F = _F
    timm = _timm
    Image = _Image
    gaussian_filter = _gaussian_filter


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
        image = image.resize(image_size[::-1], resample=Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _to_tensor_image(sample, image_size: tuple[int, int]):
    _require_dependencies()
    image = sample.image if sample.image is not None else _load_image(sample.image_path, image_size)
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image for {sample.image_id}, got shape {image.shape}")
    image = np.clip(image, 0.0, 1.0)
    return torch.from_numpy(image).permute(2, 0, 1).contiguous()


def _create_feature_extractor(
    backbone_candidates: list[str],
    out_indices: tuple[int, ...],
    image_size: tuple[int, int],
    device,
):
    _require_dependencies()

    class TimmFeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone_name, self.model = self._build_model()
            self.register_buffer(
                "mean",
                torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )
            self.register_buffer(
                "std",
                torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
                persistent=False,
            )

        def _build_model(self):
            last_error = None
            for backbone_name in backbone_candidates:
                try:
                    model = timm.create_model(
                        backbone_name,
                        pretrained=True,
                        features_only=True,
                        out_indices=out_indices,
                    ).to(device)
                    model.eval()
                    with torch.no_grad():
                        h, w = image_size
                        dummy = torch.zeros(1, 3, h, w, device=device)
                        _ = model((dummy - self._mean_like(dummy)) / self._std_like(dummy))
                    print(f"Using PatchCore backbone: {backbone_name}")
                    return backbone_name, model
                except Exception as exc:
                    print(f"Backbone '{backbone_name}' failed: {exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    last_error = exc
            raise RuntimeError(f"All PatchCore backbones failed. Last error: {last_error}")

        @staticmethod
        def _mean_like(x):
            return x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)

        @staticmethod
        def _std_like(x):
            return x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)

        def forward(self, x):
            x = (x - self.mean) / self.std
            return self.model(x)

    return TimmFeatureExtractor().to(device).eval()


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
    if n_points <= k:
        return torch.arange(n_points, dtype=torch.long)

    work = _random_project(points, projection_dim, seed)
    generator = torch.Generator(device=work.device)
    generator.manual_seed(int(seed))

    first_idx = int(torch.randint(0, n_points, (1,), generator=generator, device=work.device).item())
    selected = [first_idx]
    min_dist = torch.cdist(work, work[first_idx : first_idx + 1]).squeeze(1)

    for _ in tqdm(range(1, k), desc="Greedy coreset", leave=False):
        next_idx = int(torch.argmax(min_dist).item())
        selected.append(next_idx)
        dist_to_new = torch.cdist(work, work[next_idx : next_idx + 1]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_to_new)

    return torch.tensor(selected, dtype=torch.long)


def _knn_search(query, bank, bank_chunk_size: int):
    query = query.float()
    bank = bank.float()
    best_dist = torch.full((query.shape[0],), float("inf"), dtype=torch.float32)

    for start in range(0, bank.shape[0], bank_chunk_size):
        bank_chunk = bank[start : start + bank_chunk_size]
        distances = torch.cdist(query, bank_chunk)
        values, _ = distances.min(dim=1)
        best_dist = torch.minimum(best_dist, values.cpu())

    return best_dist


def _normalize_maps(raw_predictions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if not raw_predictions:
        return {}
    min_value = min(float(np.nanmin(x)) for x in raw_predictions.values())
    max_value = max(float(np.nanmax(x)) for x in raw_predictions.values())
    dynamic_range = max_value - min_value
    if dynamic_range <= 1e-12:
        for image_id, x in raw_predictions.items():
            raw_predictions[image_id] = np.zeros_like(x, dtype=np.float16)
        return raw_predictions

    for image_id, x in raw_predictions.items():
        normalized = (x.astype(np.float32, copy=False) - min_value) / dynamic_range
        raw_predictions[image_id] = np.clip(normalized, 0.0, 1.0).astype(np.float16)
    return raw_predictions


class _SampleDataset:
    def __init__(self, samples, image_size: tuple[int, int]):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        return {
            "image": _to_tensor_image(sample, self.image_size),
            "image_id": sample.image_id,
            "class_name": sample.class_name,
            "path": str(sample.image_path),
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
        self.out_indices = tuple(int(x) for x in method_config.get("out_indices", (2, 3)))
        self.patchsize = int(method_config.get("patchsize", 3))
        self.coreset_fraction = float(method_config.get("coreset_fraction", 0.01))
        self.candidate_pool_size = int(method_config.get("candidate_pool_size", 6000))
        max_coreset_size = method_config.get("max_coreset_size", None)
        self.max_coreset_size = int(max_coreset_size) if max_coreset_size is not None else None
        self.projection_dim = method_config.get("projection_dim", 512)
        self.bank_chunk_size = int(method_config.get("bank_chunk_size", 2048))
        self.sigma = float(method_config.get("sigma", 4.0))
        self.class_wise = bool(method_config.get("class_wise", True))

        backbone = method_config.get("backbone", "wide_resnet50_2")
        candidates = method_config.get("backbone_candidates")
        if candidates is None:
            candidates = [backbone]
        self.backbone_candidates = [str(x) for x in candidates]

        requested_device = method_config.get("device")
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        self.feature_extractor = _create_feature_extractor(
            self.backbone_candidates,
            self.out_indices,
            self.image_size,
            self.device,
        )
        self.backbone_name = self.feature_extractor.backbone_name
        self.memory_banks: dict[str, Any] = {}
        self.feature_grid_shape: tuple[int, int] | None = None

    def fit(self, train_data, val_data=None):
        clean_samples = [s for s in train_data if s.label in (None, 0)]
        if not clean_samples:
            raise ValueError("PatchCore Lite requires at least one clean training image")

        grouped = self._group_samples(clean_samples) if self.class_wise else {GLOBAL_BANK_KEY: clean_samples}
        for class_name, samples in grouped.items():
            print(f"Fitting PatchCore memory bank for {class_name}: {len(samples)} images")
            self.memory_banks[class_name] = self._fit_memory_bank(samples)
        return self

    def predict(self, test_data) -> dict[str, np.ndarray]:
        if not self.memory_banks:
            raise RuntimeError("PatchCore Lite has not been fitted yet")

        samples = list(test_data)
        grouped = self._group_samples(samples) if self.class_wise else {GLOBAL_BANK_KEY: samples}
        raw_predictions: dict[str, np.ndarray] = {}
        for class_name, class_samples in grouped.items():
            bank_key = class_name if self.class_wise else GLOBAL_BANK_KEY
            if bank_key not in self.memory_banks:
                raise RuntimeError(f"No PatchCore memory bank found for class '{class_name}'")
            raw_predictions.update(self._predict_with_bank(class_samples, self.memory_banks[bank_key]))
        return _normalize_maps(raw_predictions)

    def save(self, output_dir: str | Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "patchcore_lite.pt"
        torch.save(
            {
                "config": self.config,
                "backbone_name": self.backbone_name,
                "memory_banks": self.memory_banks,
                "feature_grid_shape": self.feature_grid_shape,
            },
            path,
        )
        return path

    def load(self, checkpoint_path: str | Path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.memory_banks = checkpoint["memory_banks"]
        self.feature_grid_shape = checkpoint.get("feature_grid_shape")
        return self

    @staticmethod
    def _group_samples(samples) -> dict[str, list]:
        grouped: dict[str, list] = {}
        for sample in samples:
            grouped.setdefault(sample.class_name, []).append(sample)
        return grouped

    def _make_loader(self, samples, shuffle: bool = False):
        dataset = _SampleDataset(samples, self.image_size)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

    def _extract_patch_embeddings(self, images):
        images = images.to(self.device, non_blocking=True)
        features = self.feature_extractor(images)
        target_hw = features[0].shape[-2:]
        processed = []

        for feat in features:
            feat = F.avg_pool2d(
                feat,
                kernel_size=self.patchsize,
                stride=1,
                padding=self.patchsize // 2,
            )
            if feat.shape[-2:] != target_hw:
                feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
            processed.append(feat)

        embeddings = torch.cat(processed, dim=1)
        embeddings = embeddings.permute(0, 2, 3, 1).contiguous()
        return embeddings

    def _fit_memory_bank(self, samples):
        self.feature_extractor.eval()
        candidate_bank = None
        candidate_keys = None
        n_total = 0

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        with torch.no_grad():
            for batch in tqdm(self._make_loader(samples), desc="PatchCore feature extraction"):
                embeddings = self._extract_patch_embeddings(batch["image"])
                _, h, w, channels = embeddings.shape
                self.feature_grid_shape = (h, w)
                flat_embeddings = embeddings.reshape(-1, channels).detach().cpu().float()
                n_total += int(flat_embeddings.shape[0])
                candidate_bank, candidate_keys = self._update_candidate_pool(
                    candidate_bank=candidate_bank,
                    candidate_keys=candidate_keys,
                    batch_embeddings=flat_embeddings,
                    generator=generator,
                )

        if candidate_bank is None:
            raise ValueError("No patch embeddings were extracted")

        n_coreset = max(1, int(round(n_total * self.coreset_fraction)))
        if self.max_coreset_size is not None:
            n_coreset = min(n_coreset, self.max_coreset_size)
        n_coreset = min(n_coreset, int(candidate_bank.shape[0]))

        selected_in_candidate = _greedy_coreset(
            candidate_bank,
            k=n_coreset,
            projection_dim=self.projection_dim,
            seed=self.seed,
        )
        memory_bank = candidate_bank[selected_in_candidate].contiguous().cpu()

        print(
            f"Seen bank: {n_total:,} patches x {candidate_bank.shape[1]} dims; "
            f"candidate pool: {candidate_bank.shape[0]:,} patches; "
            f"coreset: {memory_bank.shape[0]:,} patches"
        )
        return memory_bank

    def _update_candidate_pool(
        self,
        candidate_bank,
        candidate_keys,
        batch_embeddings,
        generator,
    ):
        pool_size = max(1, int(self.candidate_pool_size))
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

    def _predict_with_bank(self, samples, memory_bank) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        bank = memory_bank.cpu().float()

        with torch.no_grad():
            for batch in tqdm(self._make_loader(samples), desc="PatchCore inference"):
                images = batch["image"]
                embeddings = self._extract_patch_embeddings(images)
                batch_size, h, w, channels = embeddings.shape
                flat_embeddings = embeddings.reshape(-1, channels).detach().cpu().float()
                nn_distances = _knn_search(flat_embeddings, bank, self.bank_chunk_size)
                patch_scores = nn_distances.view(batch_size, h, w)

                maps = F.interpolate(
                    patch_scores.unsqueeze(1),
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

                for i, image_id in enumerate(batch["image_id"]):
                    anomaly_map = maps[i, 0].detach().cpu().numpy().astype(np.float32)
                    if self.sigma > 0:
                        anomaly_map = gaussian_filter(anomaly_map, sigma=self.sigma).astype(np.float32)
                    predictions[str(image_id)] = anomaly_map.astype(np.float16)

        return predictions
