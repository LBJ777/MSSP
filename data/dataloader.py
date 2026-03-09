"""
data/dataloader.py
------------------
Data-loading module for MSSP experiments.
Copied from DRIFT_new for standalone operation.
Supports AIGCDetectBenchmark directory layout.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image, ImageFile
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import get_transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

_REAL_DIR = "0_real"
_FAKE_DIR = "1_fake"
_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _collect_images(directory: Union[str, Path]) -> List[str]:
    directory = Path(directory)
    paths: List[str] = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if Path(fname).suffix.lower() in _SUPPORTED_EXTENSIONS:
                paths.append(str(Path(root) / fname))
    return sorted(paths)


def _discover_generators(root: Union[str, Path]) -> List[str]:
    root = Path(root)
    generators: List[str] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / _REAL_DIR).is_dir() or (entry / _FAKE_DIR).is_dir():
            generators.append(entry.name)
    return generators


class _MSSSPDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int, str]],
        transform,
        mode: str,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        img_path, binary_label, generator_name = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == "binary_mode":
            return img, binary_label
        elif self.mode == "attribution_mode":
            return img, generator_name
        else:
            raise ValueError(f"Unknown mode '{self.mode}'.")


class DRIFTDataLoader:
    """Unified data-loading interface for MSSP experiments."""

    def __init__(
        self,
        root: Union[str, Path],
        mode: str = "binary_mode",
        split: str = "train",
        image_size: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        num_samples: Optional[int] = None,
        split_ratios: Optional[Dict[str, float]] = None,
        seed: int = 42,
        pin_memory: bool = True,
        shuffle: Optional[bool] = None,
        normalize_for_adm: bool = True,
    ) -> None:
        if mode not in ("binary_mode", "attribution_mode"):
            raise ValueError(f"mode must be 'binary_mode' or 'attribution_mode', got '{mode}'.")
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'.")

        self.root = Path(root)
        self.mode = mode
        self.split = split
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_samples = num_samples
        self.seed = seed
        self.pin_memory = pin_memory

        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        _total = sum(split_ratios.values())
        if abs(_total - 1.0) > 1e-6:
            raise ValueError(f"split_ratios must sum to 1.0, got {_total:.4f}.")
        self.split_ratios = split_ratios
        self.shuffle = (split == "train") if shuffle is None else shuffle

        self._samples: List[Tuple[str, int, str]] = self._build_samples()
        self._print_stats()
        self._transform = get_transforms(
            split=split,
            image_size=image_size,
            normalize_for_adm=normalize_for_adm,
        )

    def _build_samples(self) -> List[Tuple[str, int, str]]:
        generators = _discover_generators(self.root)
        all_samples: List[Tuple[str, int, str]] = []

        if generators:
            for gen_name in generators:
                gen_dir = self.root / gen_name
                all_samples.extend(self._collect_from_generator(gen_dir, gen_name))
        else:
            all_samples.extend(self._collect_from_generator(self.root, "unknown"))

        if not all_samples:
            raise RuntimeError(
                f"No images found under '{self.root}'. "
                "Make sure the directory contains '0_real' and/or '1_fake' sub-folders."
            )

        rng = random.Random(self.seed)
        rng.shuffle(all_samples)
        return self._apply_split(all_samples)

    def _collect_from_generator(
        self,
        gen_dir: Path,
        generator_name: str,
    ) -> List[Tuple[str, int, str]]:
        samples: List[Tuple[str, int, str]] = []

        real_dir = gen_dir / _REAL_DIR
        if real_dir.is_dir():
            real_paths = _collect_images(real_dir)
            if self.num_samples is not None:
                real_paths = real_paths[: self.num_samples]
            samples.extend((p, 0, generator_name) for p in real_paths)

        fake_dir = gen_dir / _FAKE_DIR
        if fake_dir.is_dir():
            fake_paths = _collect_images(fake_dir)
            if self.num_samples is not None:
                fake_paths = fake_paths[: self.num_samples]
            samples.extend((p, 1, generator_name) for p in fake_paths)

        return samples

    def _apply_split(self, samples):
        n = len(samples)
        train_end = int(n * self.split_ratios["train"])
        val_end = train_end + int(n * self.split_ratios["val"])
        if self.split == "train":
            return samples[:train_end]
        elif self.split == "val":
            return samples[train_end:val_end]
        else:
            return samples[val_end:]

    def _print_stats(self) -> None:
        n_real = sum(1 for _, lbl, _ in self._samples if lbl == 0)
        n_fake = sum(1 for _, lbl, _ in self._samples if lbl == 1)
        generators: Dict[str, int] = {}
        for _, _, gen in self._samples:
            generators[gen] = generators.get(gen, 0) + 1
        print(
            f"[DRIFTDataLoader] split={self.split!r} | mode={self.mode!r} | "
            f"total={len(self._samples)} (real={n_real}, fake={n_fake})"
        )
        for gen, cnt in sorted(generators.items()):
            print(f"  generator={gen!r}: {cnt} samples")

    def get_dataset(self) -> _MSSSPDataset:
        return _MSSSPDataset(
            samples=self._samples,
            transform=self._transform,
            mode=self.mode,
        )

    def get_dataloader(self, **kwargs) -> DataLoader:
        dataset = self.get_dataset()
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        loader_kwargs.update(kwargs)
        return DataLoader(dataset, **loader_kwargs)

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def generator_names(self) -> List[str]:
        return sorted({gen for _, _, gen in self._samples})

    @property
    def samples(self) -> List[Tuple[str, int, str]]:
        return self._samples
