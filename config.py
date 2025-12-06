# -*- coding: utf-8 -*-
"""
Konfigürasyon dosyası - Tüm path'ler, hiperparametreler ve sabitler
"""

import os

# =============================================================================
# PATH TANIMLARI
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Train paths
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, "data")
TRAIN_LABELS_JSON = os.path.join(TRAIN_DIR, "labels.json")

# Test paths
TEST_DIR = os.path.join(DATA_DIR, "test")
TEST_IMAGES_DIR = os.path.join(TEST_DIR, "data")
TEST_LABELS_JSON = os.path.join(DATA_DIR, "test_labels_200.json")  # data/ altında

# Output paths
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")

# YOLO veri yapısı paths
YOLO_DIR = os.path.join(OUTPUT_DIR, "yolo_data")
YOLO_IMAGES_DIR = os.path.join(YOLO_DIR, "images")
YOLO_LABELS_DIR = os.path.join(YOLO_DIR, "labels")
YOLO_TRAIN_IMAGES = os.path.join(YOLO_IMAGES_DIR, "train")
YOLO_VAL_IMAGES = os.path.join(YOLO_IMAGES_DIR, "val")
YOLO_TEST_IMAGES = os.path.join(YOLO_IMAGES_DIR, "test")
YOLO_TRAIN_LABELS = os.path.join(YOLO_LABELS_DIR, "train")
YOLO_VAL_LABELS = os.path.join(YOLO_LABELS_DIR, "val")
YOLO_TEST_LABELS = os.path.join(YOLO_LABELS_DIR, "test")

# =============================================================================
# SINIF TANIMLARI (11 Parazit Türü)
# =============================================================================
CLASS_NAMES = {
    0: "Ascaris lumbricoides",
    1: "Capillaria philippinensis",
    2: "Enterobius vermicularis",
    3: "Fasciolopsis buski",
    4: "Hookworm egg",
    5: "Hymenolepis diminuta",
    6: "Hymenolepis nana",
    7: "Opisthorchis viverrine",
    8: "Paragonimus spp",
    9: "Taenia spp. egg",
    10: "Trichuris trichiura"
}

NUM_CLASSES = len(CLASS_NAMES)

# =============================================================================
# GÖRÜNTÜ İŞLEME PARAMETRELERİ
# =============================================================================
IMAGE_CONFIG = {
    "target_width": 1200,
    "target_height": 1200,
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid_size": (8, 8),
    "gaussian_kernel_size": (5, 5),
    "median_kernel_size": 5,
    "bilateral_d": 9,
    "bilateral_sigma_color": 75,
    "bilateral_sigma_space": 75
}

# =============================================================================
# YOLO HİPERPARAMETRELERİ
# =============================================================================
YOLO_CONFIG = {
    "imgsz": 640,
    "batch": 8,
    "epochs": 300,
    "workers": 4,
    "optimizer": "SGD",
    "lr0": 3e-3,
    "lrf": 2e-4,
    "momentum": 0.9,
    "weight_decay": 0.01,
    "warmup_epochs": 15,
    "cos_lr": True,
    # Augmentation
    "degrees": 45.0,
    "translate": 0.2,
    "flipud": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.3
}

# =============================================================================
# MAKİNE ÖĞRENMESİ HİPERPARAMETRELERİ
# =============================================================================
ML_CONFIG = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 20,
        "criterion": "entropy",
        "random_state": 42
    },
    "xgboost": {
        "max_depth": 6,
        "n_estimators": 200,
        "subsample": 0.8,
        "random_state": 42
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale"
    },
    "logistic_regression": {
        "penalty": "l2",
        "solver": "liblinear",
        "C": 1.0,
        "max_iter": 1000
    }
}

# =============================================================================
# EĞİTİM PARAMETRELERİ
# =============================================================================
TRAIN_CONFIG = {
    "val_split": 0.2,
    "random_state": 42,
    "device": "cuda:0"  # veya "cpu"
}

# =============================================================================
# LBP/LTP ÖZNİTELİK PARAMETRELERİ
# =============================================================================
FEATURE_CONFIG = {
    "lbp_radius": 1,
    "lbp_n_points": 8,
    "lbp_selected_features": 12,
    "ltp_selected_features": 12
}
