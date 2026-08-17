"""Compact convolutional classifier for 64x64 Quick Draw bitmaps."""

import random

import numpy as np
import torch
from torch import nn


INPUT_SHAPE = (1, 64, 64)
CHANNELS = (32, 64, 128)
EMBEDDING_SIZE = 192
DROPOUT = 0.2


def set_deterministic_seed(seed):
    """Seed Python, NumPy, and PyTorch model initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class QuickDrawCNN(nn.Module):
    """Small CNN producing unnormalized class logits."""

    def __init__(self, num_classes):
        super().__init__()
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError('num_classes must be at least 2')

        self.config = {
            'inputShape': list(INPUT_SHAPE),
            'numClasses': num_classes,
            'channels': list(CHANNELS),
            'embeddingSize': EMBEDDING_SIZE,
            'dropout': DROPOUT,
        }
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CHANNELS[2] * 4 * 4, EMBEDDING_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBEDDING_SIZE, num_classes),
        )

    def forward(self, bitmaps):
        if bitmaps.ndim != 4 or tuple(bitmaps.shape[1:]) != INPUT_SHAPE:
            raise ValueError(
                f'Expected input shape [batch, {INPUT_SHAPE[0]}, '
                f'{INPUT_SHAPE[1]}, {INPUT_SHAPE[2]}]',
            )
        return self.classifier(self.features(bitmaps))


def create_model(num_classes, seed):
    """Create a model with deterministic initial parameters."""
    set_deterministic_seed(seed)
    return QuickDrawCNN(num_classes=num_classes)
