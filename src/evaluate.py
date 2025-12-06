# -*- coding: utf-8 -*-
"""
Model değerlendirme ve metrik hesaplama
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, accuracy_score, classification_report
)
from config import CLASS_NAMES, VISUALIZATIONS_DIR


def calculate_iou(box1: list, box2: list) -> float:
    """IoU (Intersection over Union) hesapla."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0


def calculate_metrics(y_true, y_pred) -> dict:
    """Precision, Recall, F1, Accuracy hesapla."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    return metrics


def calculate_map(predictions_df: pd.DataFrame, iou_threshold: float = 0.5) -> float:
    """mAP hesapla."""
    ap_list = []
    
    for i in range(len(predictions_df)):
        pred_box = [
            predictions_df.loc[i, "x1"],
            predictions_df.loc[i, "y1"],
            predictions_df.loc[i, "x2"],
            predictions_df.loc[i, "y2"]
        ]
        gt_box = [
            predictions_df.loc[i, "gt_x1"],
            predictions_df.loc[i, "gt_y1"],
            predictions_df.loc[i, "gt_x2"],
            predictions_df.loc[i, "gt_y2"]
        ]
        
        iou = calculate_iou(pred_box, gt_box)
        ap_list.append(1 if iou >= iou_threshold else 0)
    
    return np.mean(ap_list)


def calculate_map_50_95(predictions_df: pd.DataFrame) -> float:
    """mAP@50-95 hesapla."""
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    maps = [calculate_map(predictions_df, t) for t in iou_thresholds]
    return np.mean(maps)


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """Karışıklık matrisi görselleştir."""
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    class_names = [CLASS_NAMES.get(i, f'Sınıf_{i}') for i in unique_classes]
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Karışıklık Matrisi')
    plt.ylabel('Gerçek Sınıflar')
    plt.xlabel('Tahmin Edilen Sınıflar')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(VISUALIZATIONS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Karışıklık matrisi kaydedildi: {save_path}")


def evaluate_model(y_true, y_pred, predictions_df: pd.DataFrame = None) -> dict:
    """Tam değerlendirme pipeline."""
    results = calculate_metrics(y_true, y_pred)
    
    if predictions_df is not None:
        results['mAP@50'] = calculate_map(predictions_df, 0.5)
        results['mAP@50-95'] = calculate_map_50_95(predictions_df)
    
    # Sonuçları yazdır
    print("\n=== Değerlendirme Sonuçları ===")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    
    return results


def print_classification_report(y_true, y_pred):
    """Detaylı sınıflandırma raporu."""
    unique_classes = np.unique(np.concatenate([y_true, y_pred]))
    target_names = [CLASS_NAMES.get(i, f'Sınıf_{i}') for i in unique_classes]
    
    report = classification_report(y_true, y_pred, target_names=target_names)
    print("\n=== Sınıflandırma Raporu ===")
    print(report)
    return report
