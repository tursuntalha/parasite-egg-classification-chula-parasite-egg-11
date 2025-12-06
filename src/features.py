# -*- coding: utf-8 -*-
"""
Makine öğrenmesi için öznitelik çıkarma fonksiyonları
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from config import FEATURE_CONFIG


def extract_area(mask: np.ndarray) -> int:
    """Nesnenin kapladığı piksel sayısı."""
    return np.sum(mask > 0)


def extract_mean_intensity(img: np.ndarray, mask: np.ndarray = None) -> float:
    """Piksel yoğunluklarının ortalaması."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    if mask is not None:
        return np.mean(gray[mask > 0])
    return np.mean(gray)


def extract_lbp(img: np.ndarray, n_features: int = None) -> np.ndarray:
    """
    Local Binary Pattern öznitelikleri.
    Yerel piksel desenleriyle doku tanımı.
    """
    if n_features is None:
        n_features = FEATURE_CONFIG["lbp_selected_features"]
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    radius = FEATURE_CONFIG["lbp_radius"]
    n_points = FEATURE_CONFIG["lbp_n_points"]
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
    
    # En önemli n_features özniteliği seç
    indices = np.argsort(hist)[::-1][:n_features]
    return hist[indices]


def extract_ltp(img: np.ndarray, threshold: int = 5, n_features: int = None) -> np.ndarray:
    """
    Local Ternary Pattern öznitelikleri.
    LBP'den daha hassas doku tanımı.
    """
    if n_features is None:
        n_features = FEATURE_CONFIG["ltp_selected_features"]
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    h, w = gray.shape
    ltp_upper = np.zeros_like(gray, dtype=np.uint8)
    ltp_lower = np.zeros_like(gray, dtype=np.uint8)
    
    # 3x3 komşuluk için LTP hesapla
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            center = gray[i, j]
            code_upper = 0
            code_lower = 0
            
            neighbors = [
                gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                gray[i+1, j-1], gray[i, j-1]
            ]
            
            for k, neighbor in enumerate(neighbors):
                if neighbor > center + threshold:
                    code_upper |= (1 << k)
                elif neighbor < center - threshold:
                    code_lower |= (1 << k)
            
            ltp_upper[i, j] = code_upper
            ltp_lower[i, j] = code_lower
    
    # Histogram
    hist_upper, _ = np.histogram(ltp_upper.ravel(), bins=256, density=True)
    hist_lower, _ = np.histogram(ltp_lower.ravel(), bins=256, density=True)
    combined = np.concatenate([hist_upper, hist_lower])
    
    # En önemli öznitelikleri seç
    indices = np.argsort(combined)[::-1][:n_features]
    return combined[indices]


def extract_rgb_mean(img: np.ndarray) -> np.ndarray:
    """RGB kanal ortalamalarını çıkar."""
    if len(img.shape) != 3:
        return np.array([np.mean(img)] * 3)
    
    b, g, r = cv2.split(img)
    return np.array([np.mean(r), np.mean(g), np.mean(b)])


def extract_hsv_mean(img: np.ndarray) -> np.ndarray:
    """HSV kanal ortalamalarını çıkar."""
    if len(img.shape) != 3:
        return np.array([0, 0, np.mean(img)])
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.array([np.mean(h), np.mean(s), np.mean(v)])


def extract_spectral_entropy(img: np.ndarray) -> float:
    """Frekans bileşenlerinin karmaşıklığı (spektral entropi)."""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # FFT uygula
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Normalize et ve entropi hesapla
    magnitude = magnitude / np.sum(magnitude)
    magnitude = magnitude.flatten()
    magnitude = magnitude[magnitude > 0]  # Log için sıfırları kaldır
    
    return entropy(magnitude)


def extract_perimeter(mask: np.ndarray) -> float:
    """Nesnenin çevresini hesapla."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return cv2.arcLength(contours[0], True)
    return 0.0


def extract_all_features(img: np.ndarray, mask: np.ndarray = None) -> dict:
    """Tüm öznitelikleri çıkar ve sözlük olarak döndür."""
    features = {}
    
    # Temel öznitelikler
    if mask is not None:
        features['area'] = extract_area(mask)
        features['perimeter'] = extract_perimeter(mask)
    else:
        features['area'] = img.shape[0] * img.shape[1]
        features['perimeter'] = 2 * (img.shape[0] + img.shape[1])
    
    features['mean_intensity'] = extract_mean_intensity(img, mask)
    
    # Renk öznitelikleri
    rgb_mean = extract_rgb_mean(img)
    features['rgb_mean_r'] = rgb_mean[0]
    features['rgb_mean_g'] = rgb_mean[1]
    features['rgb_mean_b'] = rgb_mean[2]
    
    hsv_mean = extract_hsv_mean(img)
    features['hsv_mean_h'] = hsv_mean[0]
    features['hsv_mean_s'] = hsv_mean[1]
    features['hsv_mean_v'] = hsv_mean[2]
    
    # Doku öznitelikleri
    lbp_features = extract_lbp(img)
    for i, val in enumerate(lbp_features):
        features[f'lbp_{i}'] = val
    
    ltp_features = extract_ltp(img)
    for i, val in enumerate(ltp_features):
        features[f'ltp_{i}'] = val
    
    # Frekans özniteliği
    features['spectral_entropy'] = extract_spectral_entropy(img)
    
    return features


def extract_features_from_dataset(image_paths: list, labels: list = None) -> tuple:
    """Veri kümesindeki tüm görüntülerden öznitelik çıkar."""
    import pandas as pd
    
    all_features = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            features = extract_all_features(img)
            all_features.append(features)
    
    df = pd.DataFrame(all_features)
    
    if labels is not None:
        df['label'] = labels
    
    return df
