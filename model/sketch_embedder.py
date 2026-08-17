"""Shared encoder mapping sketch and icon bitmaps into one embedding space."""

import math

import torch
from torch import nn

from model.quickdraw_cnn import (
    CHANNELS,
    DROPOUT,
    EMBEDDING_SIZE,
    INPUT_SHAPE,
    set_deterministic_seed,
)


DEFAULT_EMBEDDING_DIM = 64
INITIAL_LOGIT_SCALE = 10.0
MAXIMUM_LOGIT_SCALE = 100.0


def _feature_stack():
    """Rebuild the classifier backbone; quickdraw_cnn.py is hash-pinned and cannot be refactored."""
    return nn.Sequential(
        nn.Conv2d(1, CHANNELS[0], kernel_size=5, padding=2),
        nn.BatchNorm2d(CHANNELS[0]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(CHANNELS[0], CHANNELS[1], kernel_size=3, padding=1),
        nn.BatchNorm2d(CHANNELS[1]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(CHANNELS[1], CHANNELS[2], kernel_size=3, padding=1),
        nn.BatchNorm2d(CHANNELS[2]),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)),
    )


class SketchEmbedder(nn.Module):
    """Encodes a 64x64 bitmap into a unit-length embedding plus auxiliary class logits."""

    def __init__(self, num_classes, embedding_dim=DEFAULT_EMBEDDING_DIM):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be at least 2')
        if not isinstance(embedding_dim, int) or embedding_dim < 8:
            raise ValueError('embedding_dim must be at least 8')

        self.config = {
            'inputShape': list(INPUT_SHAPE),
            'numClasses': num_classes,
            'channels': list(CHANNELS),
            'hiddenSize': EMBEDDING_SIZE,
            'embeddingDim': embedding_dim,
            'dropout': DROPOUT,
            'initialLogitScale': INITIAL_LOGIT_SCALE,
        }
        self.features = _feature_stack()
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CHANNELS[2] * 4 * 4, EMBEDDING_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBEDDING_SIZE, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        # Logits off a unit-length embedding are bounded by the weight norm, so without a
        # learnable scale cross-entropy cannot sharpen and the weakest classes never separate.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(INITIAL_LOGIT_SCALE)))

    def embed(self, bitmaps):
        if bitmaps.ndim != 4 or tuple(bitmaps.shape[1:]) != INPUT_SHAPE:
            raise ValueError(
                f'Expected input shape [batch, {INPUT_SHAPE[0]}, '
                f'{INPUT_SHAPE[1]}, {INPUT_SHAPE[2]}]',
            )
        projected = self.projection(self.features(bitmaps))
        return torch.nn.functional.normalize(projected, p=2.0, dim=1)

    def forward(self, bitmaps):
        embedding = self.embed(bitmaps)
        scale = self.logit_scale.exp().clamp(max=MAXIMUM_LOGIT_SCALE)
        return embedding, self.classifier(embedding) * scale


def create_embedder(num_classes, seed, embedding_dim=DEFAULT_EMBEDDING_DIM):
    """Create an embedder with deterministic initial parameters."""
    set_deterministic_seed(seed)
    return SketchEmbedder(num_classes=num_classes, embedding_dim=embedding_dim)
