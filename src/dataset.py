# -*- coding: utf-8 -*-
"""
Veri kümesi hazırlama ve YOLO formatına dönüştürme
"""

import os
import json
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TRAIN_IMAGES_DIR, TRAIN_LABELS_JSON, TEST_IMAGES_DIR, TEST_LABELS_JSON,
    YOLO_TRAIN_IMAGES, YOLO_VAL_IMAGES, YOLO_TEST_IMAGES,
    YOLO_TRAIN_LABELS, YOLO_VAL_LABELS, YOLO_TEST_LABELS,
    YOLO_DIR, NUM_CLASSES, CLASS_NAMES, TRAIN_CONFIG, IMAGE_CONFIG
)
from .preprocessing import preprocess_image


def load_json_labels(json_path: str) -> dict:
    """JSON dosyasından etiketleri yükle."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def merge_annotations(data: dict) -> pd.DataFrame:
    """Images ve annotations'ı birleştir."""
    images_df = pd.DataFrame(data['images'])
    annotations_df = pd.DataFrame(data['annotations'])
    
    merged_df = images_df.merge(
        annotations_df[['image_id', 'category_id', 'bbox']],
        left_on='id',
        right_on='image_id',
        how='left'
    )
    return merged_df[['file_name', 'category_id', 'bbox']].drop_duplicates(
        subset='file_name', keep='first'
    )


def save_yolo_labels(output_dir: str, image_name: str, bboxes: list, 
                     category_ids: list, img_width: int, img_height: int):
    """YOLO formatında etiket dosyası kaydet."""
    label_path = os.path.join(output_dir, image_name.replace(".jpg", ".txt"))
    with open(label_path, 'w') as f:
        for bbox, category_id in zip(bboxes, category_ids):
            x, y, w, h = bbox
            center_x = (x + w / 2) / img_width
            center_y = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            f.write(f"{category_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")


def resize_bboxes(bboxes: list, old_width: int, old_height: int, 
                  new_width: int, new_height: int) -> list:
    """Bounding box koordinatlarını yeniden hesapla."""
    resized = []
    for bbox in bboxes:
        x, y, w, h = bbox
        new_x = x * new_width / old_width
        new_y = y * new_height / old_height
        new_w = w * new_width / old_width
        new_h = h * new_height / old_height
        resized.append([new_x, new_y, new_w, new_h])
    return resized


def create_data_yaml(output_path: str, train_path: str, val_path: str, 
                     test_path: str = None):
    """YOLO için data.yaml dosyası oluştur."""
    content = f"""train: {train_path}
val: {val_path}
"""
    if test_path:
        content += f"test: {test_path}\n"
    
    content += f"\nnc: {NUM_CLASSES}\n"
    content += f"names: {list(CLASS_NAMES.values())}\n"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"data.yaml oluşturuldu: {output_path}")


def prepare_train_data(val_split: float = None):
    """Eğitim verisini hazırla ve train/val olarak ayır."""
    if val_split is None:
        val_split = TRAIN_CONFIG["val_split"]
    
    # JSON yükle
    data = load_json_labels(TRAIN_LABELS_JSON)
    df = merge_annotations(data)
    
    # Train/Val split
    train_df, val_df = train_test_split(
        df, 
        test_size=val_split, 
        random_state=TRAIN_CONFIG["random_state"]
    )
    
    target_w = IMAGE_CONFIG["target_width"]
    target_h = IMAGE_CONFIG["target_height"]
    
    # Train verisi işle
    for _, row in train_df.iterrows():
        _process_single_image(
            row, TRAIN_IMAGES_DIR, YOLO_TRAIN_IMAGES, YOLO_TRAIN_LABELS,
            target_w, target_h
        )
    
    # Val verisi işle
    for _, row in val_df.iterrows():
        _process_single_image(
            row, TRAIN_IMAGES_DIR, YOLO_VAL_IMAGES, YOLO_VAL_LABELS,
            target_w, target_h
        )
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)} görüntü işlendi.")
    return train_df, val_df


def prepare_test_data():
    """Test verisini hazırla."""
    data = load_json_labels(TEST_LABELS_JSON)
    df = merge_annotations(data)
    
    target_w = IMAGE_CONFIG["target_width"]
    target_h = IMAGE_CONFIG["target_height"]
    
    for _, row in df.iterrows():
        _process_single_image(
            row, TEST_IMAGES_DIR, YOLO_TEST_IMAGES, YOLO_TEST_LABELS,
            target_w, target_h
        )
    
    print(f"Test: {len(df)} görüntü işlendi.")
    return df


def _process_single_image(row, source_dir: str, dest_img_dir: str, 
                          dest_label_dir: str, target_w: int, target_h: int):
    """Tek bir görüntüyü işle ve kaydet."""
    file_name = row['file_name']
    bboxes = [row['bbox']]
    category_ids = [row['category_id']]
    
    source_path = os.path.join(source_dir, file_name)
    dest_path = os.path.join(dest_img_dir, file_name)
    
    if not os.path.exists(source_path):
        return
    
    img = cv2.imread(source_path)
    if img is None:
        return
    
    old_h, old_w = img.shape[:2]
    
    # Ön işleme
    processed_img = preprocess_image(img, target_w, target_h)
    new_h, new_w = processed_img.shape[:2]
    
    # Bbox'ları yeniden hesapla
    resized_bboxes = resize_bboxes(bboxes, old_w, old_h, new_w, new_h)
    
    # Kaydet
    cv2.imwrite(dest_path, processed_img)
    save_yolo_labels(dest_label_dir, file_name, resized_bboxes, category_ids, new_w, new_h)


def prepare_all_data():
    """Tüm veriyi hazırla ve data.yaml oluştur."""
    from .utils import setup_directories
    setup_directories()
    
    prepare_train_data()
    prepare_test_data()
    
    # data.yaml oluştur
    yaml_path = os.path.join(YOLO_DIR, "data.yaml")
    create_data_yaml(
        yaml_path,
        YOLO_TRAIN_IMAGES,
        YOLO_VAL_IMAGES,
        YOLO_TEST_IMAGES
    )
    
    return yaml_path
