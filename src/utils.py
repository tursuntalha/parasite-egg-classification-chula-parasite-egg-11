# -*- coding: utf-8 -*-
"""
Yardımcı fonksiyonlar
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import torch
import pandas as pd
from config import (
    OUTPUT_DIR, MODELS_DIR, PREDICTIONS_DIR, VISUALIZATIONS_DIR,
    YOLO_DIR, YOLO_IMAGES_DIR, YOLO_LABELS_DIR,
    YOLO_TRAIN_IMAGES, YOLO_VAL_IMAGES, YOLO_TEST_IMAGES,
    YOLO_TRAIN_LABELS, YOLO_VAL_LABELS, YOLO_TEST_LABELS
)


def set_seed(seed: int = 42):
    """Reproducibility için seed ayarla."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_directories():
    """Gerekli klasör yapısını oluştur."""
    directories = [
        OUTPUT_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
        VISUALIZATIONS_DIR,
        YOLO_DIR,
        YOLO_IMAGES_DIR,
        YOLO_LABELS_DIR,
        YOLO_TRAIN_IMAGES,
        YOLO_VAL_IMAGES,
        YOLO_TEST_IMAGES,
        YOLO_TRAIN_LABELS,
        YOLO_VAL_LABELS,
        YOLO_TEST_LABELS
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Klasör yapısı oluşturuldu.")


def save_model(model, filepath: str):
    """Model kaydet."""
    if hasattr(model, 'save'):
        model.save(filepath)
    else:
        torch.save(model.state_dict(), filepath)
    print(f"Model kaydedildi: {filepath}")


def load_model(model_class, filepath: str, **kwargs):
    """Model yükle."""
    if filepath.endswith('.pt'):
        from ultralytics import YOLO
        return YOLO(filepath)
    else:
        model = model_class(**kwargs)
        model.load_state_dict(torch.load(filepath))
        return model


def save_predictions_csv(predictions: list, filepath: str):
    """Tahminleri CSV'ye kaydet."""
    df = pd.DataFrame(predictions)
    df.to_csv(filepath, index=False)
    print(f"Tahminler kaydedildi: {filepath}")


def load_predictions_csv(filepath: str) -> pd.DataFrame:
    """CSV'den tahminleri oku."""
    return pd.read_csv(filepath)


def get_device():
    """Kullanılabilir device'ı döndür."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")
