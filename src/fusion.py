# -*- coding: utf-8 -*-
"""
Model füzyonu - Derin öğrenme ve makine öğrenmesi çıktılarını birleştirme
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from config import NUM_CLASSES


def get_dl_predictions(model, image_path: str) -> np.ndarray:
    """
    Derin öğrenme modelinden olasılık tahminleri al.
    
    Returns:
        np.ndarray: (NUM_CLASSES,) boyutunda olasılık vektörü
    """
    results = model.predict(source=image_path, verbose=False)
    
    if len(results) > 0 and results[0].boxes is not None:
        # En yüksek confidence'a sahip tahmini al
        boxes = results[0].boxes
        if len(boxes) > 0:
            probs = np.zeros(NUM_CLASSES)
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            
            for c, confidence in zip(cls, conf):
                if c < NUM_CLASSES:
                    probs[c] = max(probs[c], confidence)
            
            # Normalize et
            if probs.sum() > 0:
                probs = probs / probs.sum()
            return probs
    
    return np.ones(NUM_CLASSES) / NUM_CLASSES


def get_ml_predictions(model, features: np.ndarray, scaler=None) -> np.ndarray:
    """
    Makine öğrenmesi modelinden olasılık tahminleri al.
    
    Returns:
        np.ndarray: (NUM_CLASSES,) boyutunda olasılık vektörü
    """
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1))
    else:
        features = features.reshape(1, -1)
    
    probs = model.predict_proba(features)
    return probs[0]


def late_fusion(dl_probs: np.ndarray, ml_probs: np.ndarray, 
                dl_weight: float = 0.5) -> np.ndarray:
    """
    Geç füzyon - Olasılıkların ağırlıklı ortalaması.
    
    Args:
        dl_probs: Derin öğrenme olasılıkları
        ml_probs: Makine öğrenmesi olasılıkları
        dl_weight: Derin öğrenme ağırlığı (0-1 arası)
    
    Returns:
        np.ndarray: Füzyon sonucu olasılık vektörü
    """
    ml_weight = 1.0 - dl_weight
    fused = dl_weight * dl_probs + ml_weight * ml_probs
    return fused


def fuse_predictions(dl_model, ml_model, image_path: str, features: np.ndarray,
                     scaler=None, dl_weight: float = 0.5) -> tuple:
    """
    İki modelin tahminlerini birleştir.
    
    Returns:
        tuple: (predicted_class, confidence, fused_probs)
    """
    # Tahminleri al
    dl_probs = get_dl_predictions(dl_model, image_path)
    ml_probs = get_ml_predictions(ml_model, features, scaler)
    
    # Füzyon
    fused_probs = late_fusion(dl_probs, ml_probs, dl_weight)
    
    # En yüksek olasılıklı sınıf
    predicted_class = np.argmax(fused_probs)
    confidence = fused_probs[predicted_class]
    
    return predicted_class, confidence, fused_probs


def batch_fusion(dl_model, ml_model, image_paths: list, features_df,
                 scaler=None, dl_weight: float = 0.5) -> list:
    """
    Batch halinde füzyon tahminleri yap.
    
    Returns:
        list: Her görüntü için (predicted_class, confidence) tuple'ları
    """
    results = []
    
    for i, image_path in enumerate(image_paths):
        features = features_df.iloc[i].values
        pred_class, conf, _ = fuse_predictions(
            dl_model, ml_model, image_path, features, scaler, dl_weight
        )
        results.append({
            'image_path': image_path,
            'predicted_class': pred_class,
            'confidence': conf
        })
    
    return results
