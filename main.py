# -*- coding: utf-8 -*-
"""
Ana çalıştırma scripti
"""

import argparse
import os
import pandas as pd
from config import (
    YOLO_DIR, MODELS_DIR, PREDICTIONS_DIR, 
    TRAIN_IMAGES_DIR, TEST_IMAGES_DIR, TRAIN_LABELS_JSON
)
from src.utils import setup_directories, set_seed
from src.dataset import prepare_all_data, load_json_labels, merge_annotations
from src.features import extract_features_from_dataset
from src.train import train_yolo, train_ml_models, train_fusion_model
from src.evaluate import evaluate_model, plot_confusion_matrix, print_classification_report
from src.fusion import batch_fusion
from src.ml_models import load_ml_model
from ultralytics import YOLO


def prepare_data():
    """Veri hazırlama."""
    print("Veri hazırlanıyor...")
    yaml_path = prepare_all_data()
    print(f"Veri hazırlandı. YAML: {yaml_path}")
    return yaml_path


def train_deep_learning(args):
    """Derin öğrenme modeli eğitimi."""
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    
    if not os.path.exists(yaml_path):
        yaml_path = prepare_data()
    
    model, results = train_yolo(
        data_yaml=yaml_path,
        model_type=args.model_type,
        backbone=args.backbone,
        neck=args.neck,
        epochs=args.epochs
    )
    return model


def train_machine_learning(args):
    """Makine öğrenmesi modeli eğitimi."""
    data = load_json_labels(TRAIN_LABELS_JSON)
    df = merge_annotations(data)
    
    image_paths = [os.path.join(TRAIN_IMAGES_DIR, fn) for fn in df['file_name']]
    labels = df['category_id'].values
    
    print("Öznitelikler çıkarılıyor...")
    features_df = extract_features_from_dataset(image_paths, labels)
    
    X = features_df.drop('label', axis=1).values
    y = features_df['label'].values
    
    models = train_ml_models(X, y, use_grid_search=args.grid_search)
    return models


def train_all(args):
    """Tüm modelleri eğit (füzyon için)."""
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    
    if not os.path.exists(yaml_path):
        yaml_path = prepare_data()
    
    data = load_json_labels(TRAIN_LABELS_JSON)
    df = merge_annotations(data)
    
    image_paths = [os.path.join(TRAIN_IMAGES_DIR, fn) for fn in df['file_name']]
    labels = df['category_id'].values
    
    print("Öznitelikler çıkarılıyor...")
    features_df = extract_features_from_dataset(image_paths, labels)
    
    X = features_df.drop('label', axis=1).values
    y = features_df['label'].values
    
    dl_model, ml_models, results = train_fusion_model(
        yaml_path, X, y,
        model_type=args.model_type,
        backbone=args.backbone,
        neck=args.neck,
        epochs=args.epochs
    )
    return dl_model, ml_models


def test_model(args):
    """Model test et."""
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "yolo11m_vit_fpn_e300.pt")
    
    model = YOLO(model_path)
    
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    results = model.val(data=yaml_path, imgsz=640, batch=8)
    
    print(results)
    return results


def run_fusion(args):
    """Füzyon modeli ile tahmin."""
    from config import TEST_LABELS_JSON, TEST_IMAGES_DIR
    
    dl_model_path = args.dl_model_path
    if dl_model_path is None:
        dl_model_path = os.path.join(MODELS_DIR, "yolo11m_vit_fpn_e300.pt")
    
    dl_model = YOLO(dl_model_path)
    ml_model = load_ml_model(args.ml_model)
    scaler = load_ml_model('scaler')
    
    data = load_json_labels(TEST_LABELS_JSON)
    df = merge_annotations(data)
    
    image_paths = [os.path.join(TEST_IMAGES_DIR, fn) for fn in df['file_name']]
    
    features_df = extract_features_from_dataset(image_paths)
    
    results = batch_fusion(
        dl_model, ml_model, image_paths, features_df,
        scaler, dl_weight=args.dl_weight
    )
    
    results_df = pd.DataFrame(results)
    save_path = os.path.join(PREDICTIONS_DIR, "fusion_predictions.csv")
    results_df.to_csv(save_path, index=False)
    print(f"Tahminler kaydedildi: {save_path}")
    
    return results


def run_gradcam(args):
    """GRAD-CAM görselleştirmesi oluştur."""
    from src.gradcam import generate_gradcam_visualization, batch_gradcam
    from config import VISUALIZATIONS_DIR, TEST_LABELS_JSON, TEST_IMAGES_DIR
    
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "yolo11m_vit_fpn_e300.pt")
    
    model = YOLO(model_path)
    
    try:
        target_layer = model.model.model[-2]
    except:
        print("Uyarı: Hedef katman bulunamadı, varsayılan kullanılıyor.")
        target_layer = None
    
    if args.image_path:
        save_path = os.path.join(VISUALIZATIONS_DIR, 'gradcam_output.png')
        generate_gradcam_visualization(model.model, args.image_path, target_layer, save_path=save_path)
    else:
        data = load_json_labels(TEST_LABELS_JSON)
        df = merge_annotations(data)
        
        image_paths = [os.path.join(TEST_IMAGES_DIR, fn) for fn in df['file_name'][:10]]
        batch_gradcam(model.model, image_paths, target_layer)
    
    print(f"GRAD-CAM görselleştirmeleri kaydedildi: {VISUALIZATIONS_DIR}")



def main():
    parser = argparse.ArgumentParser(description='Parazit Yumurtası Sınıflandırma')
    
    subparsers = parser.add_subparsers(dest='mode', help='Çalışma modu')
    
    # Veri hazırlama
    parser_data = subparsers.add_parser('prepare', help='Veri hazırla')
    
    # DL eğitim
    parser_dl = subparsers.add_parser('train_dl', help='Derin öğrenme eğitimi')
    parser_dl.add_argument('--model_type', type=str, default='yolo11m')
    parser_dl.add_argument('--backbone', type=str, default='vit')
    parser_dl.add_argument('--neck', type=str, default='fpn')
    parser_dl.add_argument('--epochs', type=int, default=300)
    
    # ML eğitim
    parser_ml = subparsers.add_parser('train_ml', help='Makine öğrenmesi eğitimi')
    parser_ml.add_argument('--grid_search', action='store_true')
    
    # Tüm eğitim
    parser_all = subparsers.add_parser('train_all', help='Tüm modelleri eğit')
    parser_all.add_argument('--model_type', type=str, default='yolo11m')
    parser_all.add_argument('--backbone', type=str, default='vit')
    parser_all.add_argument('--neck', type=str, default='fpn')
    parser_all.add_argument('--epochs', type=int, default=300)
    
    # Test
    parser_test = subparsers.add_parser('test', help='Model test et')
    parser_test.add_argument('--model_path', type=str, default=None)
    
    # Füzyon
    parser_fusion = subparsers.add_parser('fusion', help='Füzyon tahmin')
    parser_fusion.add_argument('--dl_model_path', type=str, default=None)
    parser_fusion.add_argument('--ml_model', type=str, default='xgboost')
    parser_fusion.add_argument('--dl_weight', type=float, default=0.5)
    
    # GRAD-CAM
    parser_gradcam = subparsers.add_parser('gradcam', help='GRAD-CAM görselleştirme')
    parser_gradcam.add_argument('--model_path', type=str, default=None)
    parser_gradcam.add_argument('--image_path', type=str, default=None, help='Tek görüntü için yol')
    
    args = parser.parse_args()
    
    # Seed ayarla
    set_seed(42)
    
    # Klasörleri oluştur
    setup_directories()
    
    if args.mode == 'prepare':
        prepare_data()
    elif args.mode == 'train_dl':
        train_deep_learning(args)
    elif args.mode == 'train_ml':
        train_machine_learning(args)
    elif args.mode == 'train_all':
        train_all(args)
    elif args.mode == 'test':
        test_model(args)
    elif args.mode == 'fusion':
        run_fusion(args)
    elif args.mode == 'gradcam':
        run_gradcam(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
