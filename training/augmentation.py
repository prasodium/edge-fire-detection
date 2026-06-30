"""Albumentations augmentation pipeline for fire/smoke detection training.

Implements every augmentation requested in the project spec. Two pipelines
are exposed: `build_train_transform` (heavy, stochastic) and
`build_val_transform` (resize/normalize only, deterministic).
"""
from __future__ import annotations

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


def _smoke_overlay(image: np.ndarray, **kwargs) -> np.ndarray:
    """Synthetic semi-transparent gray smoke patch - cheap stand-in for a full
    smoke simulator, teaches the model partial-occlusion-by-smoke appearance
    without needing a separate rendered-smoke asset library."""
    if np.random.rand() > 0.3:
        return image
    h, w = image.shape[:2]
    overlay = image.copy()
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    radius = np.random.randint(min(h, w) // 8, min(h, w) // 3)
    color = (180, 180, 180)
    cv2.circle(overlay, (cx, cy), radius, color, -1)
    alpha = np.random.uniform(0.15, 0.45)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def _shadow_simulation(image: np.ndarray, **kwargs) -> np.ndarray:
    """Random polygonal darkened region - simulates hard shadows cast across
    a seminar hall from stage equipment, podiums, or window light."""
    if np.random.rand() > 0.3:
        return image
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(
        [[np.random.randint(0, w), np.random.randint(0, h)] for _ in range(4)]
    )
    cv2.fillPoly(mask, [pts], 255)
    shadow = image.astype(np.float32)
    factor = np.random.uniform(0.4, 0.7)
    shadow[mask > 0] *= factor
    return shadow.clip(0, 255).astype(np.uint8)


def build_train_transform(image_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.6, 1.0), ratio=(0.8, 1.25), p=0.5),
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.4),
            A.Perspective(scale=(0.02, 0.08), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(),
                ],
                p=0.3,
            ),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=7),
                    A.GaussianBlur(blur_limit=5),
                    A.Defocus(radius=(1, 3)),
                ],
                p=0.3,
            ),
            A.ImageCompression(quality_lower=40, quality_upper=90, p=0.4),
            A.CoarseDropout(max_holes=4, max_height=image_size // 8, max_width=image_size // 8, p=0.3),  # occlusion
            A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.2),  # lens distortion
            A.Lambda(image=_shadow_simulation, p=1.0),
            A.Lambda(image=_smoke_overlay, p=1.0),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.2),
    )


def build_val_transform(image_size: int = 640) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
