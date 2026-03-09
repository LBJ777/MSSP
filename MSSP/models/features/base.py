"""
models/features/base.py
-----------------------
Abstract base class for MSSP feature extraction schemes.
Copied from DRIFT_new for standalone operation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class FeatureExtractor(ABC):
    """Abstract base class for feature extraction schemes."""

    @abstractmethod
    def extract(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        raise NotImplementedError

    def __call__(
        self,
        x_T: torch.Tensor,
        intermediates: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        return self.extract(x_T, intermediates)

    def validate_output(self, features: torch.Tensor) -> None:
        if features.ndim != 2:
            raise ValueError(
                f"FeatureExtractor.extract() must return a 2-D tensor "
                f"[B, feature_dim], got shape {tuple(features.shape)}."
            )
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Output feature dim mismatch: expected {self.feature_dim}, "
                f"got {features.shape[1]}."
            )

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        try:
            dim = self.feature_dim
        except NotImplementedError:
            dim = "?"
        return f"{cls}(feature_dim={dim})"
