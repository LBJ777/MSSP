"""
models/heads/binary.py
----------------------
Binary deepfake detection head for MSSP.
Copied from DRIFT_new for standalone operation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BinaryDetectionHead(nn.Module):
    """Fully-connected binary classification head."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, "
                f"got {features.shape[1]}."
            )
        return self.net(features)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(features)
        return torch.sigmoid(logits)

    def __repr__(self) -> str:
        return (
            f"BinaryDetectionHead("
            f"feature_dim={self.feature_dim}, "
            f"hidden_dim={self.hidden_dim}"
            f")"
        )
