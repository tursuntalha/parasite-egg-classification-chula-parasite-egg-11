# -*- coding: utf-8 -*-
"""
Görüntü ön işleme fonksiyonları
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from config import IMAGE_CONFIG


def preprocess_image(img: np.ndarray, target_width: int = None, target_height: int = None) -> np.ndarray:
    """
    Görüntüyü ön işlemden geçirir:
    1. HSV dönüşümü (renkli görüntüler için)
    2. Gaussian gürültü azaltma
    3. CLAHE ile kontrast iyileştirme
    4. Yeniden boyutlandırma
    """
    if target_width is None:
        target_width = IMAGE_CONFIG["target_width"]
    if target_height is None:
        target_height = IMAGE_CONFIG["target_height"]

    if len(img.shape) == 3:
        # RGB görüntü - HSV uzayında işle
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Gaussian blur (V kanalına)
        v = cv2.GaussianBlur(v, IMAGE_CONFIG["gaussian_kernel_size"], 0)

        # CLAHE (V kanalına)
        clahe = cv2.createCLAHE(
            clipLimit=IMAGE_CONFIG["clahe_clip_limit"],
            tileGridSize=IMAGE_CONFIG["clahe_tile_grid_size"]
        )
        v = clahe.apply(v)

        # Kanalları birleştir
        hsv = cv2.merge([h, s, v])
        processed_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        # Gri tonlamalı görüntü
        processed_img = cv2.GaussianBlur(img, IMAGE_CONFIG["gaussian_kernel_size"], 0)
        clahe = cv2.createCLAHE(
            clipLimit=IMAGE_CONFIG["clahe_clip_limit"],
            tileGridSize=IMAGE_CONFIG["clahe_tile_grid_size"]
        )
        processed_img = clahe.apply(processed_img)

    # Yeniden boyutlandır
    processed_img = cv2.resize(processed_img, (target_width, target_height))
    return processed_img


def apply_median_filter(img: np.ndarray, kernel_size: int = None) -> np.ndarray:
    """Tuz-biber gürültüsü giderme için median filtre."""
    if kernel_size is None:
        kernel_size = IMAGE_CONFIG["median_kernel_size"]
    return cv2.medianBlur(img, kernel_size)


def apply_bilateral_filter(img: np.ndarray, d: int = None, 
                           sigma_color: int = None, sigma_space: int = None) -> np.ndarray:
    """Kenar koruyucu bilateral filtre."""
    if d is None:
        d = IMAGE_CONFIG["bilateral_d"]
    if sigma_color is None:
        sigma_color = IMAGE_CONFIG["bilateral_sigma_color"]
    if sigma_space is None:
        sigma_space = IMAGE_CONFIG["bilateral_sigma_space"]
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Sadece CLAHE uygula."""
    if len(img.shape) == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(
            clipLimit=IMAGE_CONFIG["clahe_clip_limit"],
            tileGridSize=IMAGE_CONFIG["clahe_tile_grid_size"]
        )
        v = clahe.apply(v)
        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    else:
        clahe = cv2.createCLAHE(
            clipLimit=IMAGE_CONFIG["clahe_clip_limit"],
            tileGridSize=IMAGE_CONFIG["clahe_tile_grid_size"]
        )
        return clahe.apply(img)


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """Tuz-biber gürültüsü ekle."""
    noisy = img.copy()
    h, w = img.shape[:2]
    num_salt = int(amount * h * w / 2)
    num_pepper = int(amount * h * w / 2)

    # Salt (beyaz)
    coords = [np.random.randint(0, i, num_salt) for i in [h, w]]
    noisy[coords[0], coords[1]] = 255

    # Pepper (siyah)
    coords = [np.random.randint(0, i, num_pepper) for i in [h, w]]
    noisy[coords[0], coords[1]] = 0

    return noisy


def add_fog_effect(img: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Sis efekti ekle."""
    fog = np.ones_like(img) * 255
    return cv2.addWeighted(img, 1 - intensity, fog, intensity, 0)


def flip_horizontal(img: np.ndarray) -> np.ndarray:
    """Yatay çevirme."""
    return cv2.flip(img, 1)


def flip_vertical(img: np.ndarray) -> np.ndarray:
    """Dikey çevirme."""
    return cv2.flip(img, 0)


def create_augmented_image(img: np.ndarray, dataset_type: int = 1) -> np.ndarray:
    """
    Veri kümesi tipine göre augmentation uygula.
    
    dataset_type:
        1: Orijinal (sadece preprocess)
        2: CLAHE + Tuz-Biber + Sis
        3: Tuz-Biber + Sis + CLAHE + Yatay Çevirme + Dikey Çevirme
    """
    if dataset_type == 1:
        return preprocess_image(img)
    
    elif dataset_type == 2:
        processed = apply_clahe(img)
        processed = add_salt_pepper_noise(processed)
        processed = add_fog_effect(processed)
        return cv2.resize(processed, (IMAGE_CONFIG["target_width"], IMAGE_CONFIG["target_height"]))
    
    elif dataset_type == 3:
        processed = add_salt_pepper_noise(img)
        processed = add_fog_effect(processed)
        processed = apply_clahe(processed)
        processed = flip_horizontal(processed)
        processed = flip_vertical(processed)
        return cv2.resize(processed, (IMAGE_CONFIG["target_width"], IMAGE_CONFIG["target_height"]))
    
    return preprocess_image(img)
