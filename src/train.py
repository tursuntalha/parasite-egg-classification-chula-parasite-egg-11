# -*- coding: utf-8 -*-
"""
Eğitim scriptleri
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config import YOLO_CONFIG, YOLO_DIR, MODELS_DIR, TRAIN_CONFIG
from .models import get_model, get_base_yolo
from .ml_models import train_all_ml_models, save_ml_model
from .utils import setup_directories, get_device


def train_yolo(data_yaml: str = None, model_type: str = 'yolo11m',
               backbone: str = None, neck: str = None, epochs: int = None):
    """
    YOLO modelini eğit.
    
    Args:
        data_yaml: data.yaml dosya yolu
        model_type: 'yolo11s', 'yolo11m', 'yolo11x'
        backbone: None veya 'vit'
        neck: None, 'fpn', 'afpn'
        epochs: Epoch sayısı
    """
    setup_directories()
    
    if data_yaml is None:
        data_yaml = os.path.join(YOLO_DIR, "data.yaml")
    
    if epochs is None:
        epochs = YOLO_CONFIG["epochs"]
    
    # Model oluştur
    model = get_model(model_type, backbone, neck)
    
    # Device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Eğitim
    results = model.train(
        data=data_yaml,
        imgsz=YOLO_CONFIG["imgsz"],
        batch=YOLO_CONFIG["batch"],
        epochs=epochs,
        workers=YOLO_CONFIG["workers"],
        device=device,
        optimizer=YOLO_CONFIG["optimizer"],
        lr0=YOLO_CONFIG["lr0"],
        lrf=YOLO_CONFIG["lrf"],
        momentum=YOLO_CONFIG["momentum"],
        weight_decay=YOLO_CONFIG["weight_decay"],
        warmup_epochs=YOLO_CONFIG["warmup_epochs"],
        cos_lr=YOLO_CONFIG["cos_lr"],
        degrees=YOLO_CONFIG["degrees"],
        translate=YOLO_CONFIG["translate"],
        flipud=YOLO_CONFIG["flipud"],
        fliplr=YOLO_CONFIG["fliplr"],
        mosaic=YOLO_CONFIG["mosaic"],
        mixup=YOLO_CONFIG["mixup"]
    )
    
    # Modeli kaydet
    model_name = f"{model_type}"
    if backbone:
        model_name += f"_{backbone}"
    if neck:
        model_name += f"_{neck}"
    model_name += f"_e{epochs}"
    
    save_path = os.path.join(MODELS_DIR, f"{model_name}.pt")
    model.save(save_path)
    print(f"Model kaydedildi: {save_path}")
    
    return model, results


def train_ml_models(X_train, y_train, use_grid_search: bool = False):
    """
    Tüm makine öğrenmesi modellerini eğit.
    
    Args:
        X_train: Öznitelik matrisi
        y_train: Etiketler
        use_grid_search: Grid search kullanılsın mı
    """
    setup_directories()
    
    models = train_all_ml_models(X_train, y_train, use_grid_search)
    
    # Modelleri kaydet
    for name, model in models.items():
        save_ml_model(model, name)
    
    return models


def train_fusion_model(data_yaml: str, X_train, y_train,
                       model_type: str = 'yolo11m', backbone: str = 'vit',
                       neck: str = 'fpn', epochs: int = 200):
    """
    Füzyon modeli için hem DL hem ML modellerini eğit.
    """
    # YOLO eğit
    dl_model, dl_results = train_yolo(data_yaml, model_type, backbone, neck, epochs)
    
    # ML modelleri eğit
    ml_models = train_ml_models(X_train, y_train, use_grid_search=True)
    
    return dl_model, ml_models, dl_results
