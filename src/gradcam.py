# -*- coding: utf-8 -*-
"""
GRAD-CAM görselleştirme - Model kararlarının yorumlanabilirliği
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from config import VISUALIZATIONS_DIR, CLASS_NAMES


class GradCAM:
    """GRAD-CAM (Gradient-weighted Class Activation Mapping) implementasyonu."""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook'ları kaydet
        self._register_hooks()
    
    def _register_hooks(self):
        """Forward ve backward hook'ları kaydet."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        GRAD-CAM ısı haritası oluştur.
        
        Args:
            input_tensor: Model girdisi (1, C, H, W)
            target_class: Hedef sınıf (None ise en yüksek olasılıklı sınıf)
        
        Returns:
            np.ndarray: Isı haritası (H, W)
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Gradyan ağırlıkları
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Ağırlıklı aktivasyon haritası
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def apply_gradcam_to_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    GRAD-CAM ısı haritasını görüntüye uygula.
    
    Args:
        image: Orijinal görüntü (H, W, C)
        cam: GRAD-CAM haritası (H, W)
        alpha: Karıştırma oranı
    
    Returns:
        np.ndarray: Overlay görüntü
    """
    # CAM'i görüntü boyutuna yeniden boyutlandır
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    
    # Isı haritası oluştur
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = np.float32(heatmap) * alpha + np.float32(image) * (1 - alpha)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    
    return overlay


def generate_gradcam_visualization(model, image_path: str, target_layer, 
                                   target_class: int = None, save_path: str = None):
    """
    Tek bir görüntü için GRAD-CAM görselleştirmesi oluştur.
    
    Args:
        model: PyTorch modeli
        image_path: Görüntü dosya yolu
        target_layer: Hedef katman
        target_class: Hedef sınıf (opsiyonel)
        save_path: Kayıt yolu (opsiyonel)
    """
    # Görüntüyü yükle
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Tensor'a dönüştür
    input_tensor = preprocess_for_gradcam(image)
    
    # GRAD-CAM oluştur
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(input_tensor, target_class)
    
    # Overlay oluştur
    overlay = apply_gradcam_to_image(image, cam)
    
    # Isı haritası
    cam_resized = cv2.resize(cam, (image.shape[1], image.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Görselleştir
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Orijinal Görüntü')
    axes[0].axis('off')
    
    axes[1].imshow(overlay)
    axes[1].set_title('GRAD-CAM Overlay')
    axes[1].axis('off')
    
    axes[2].imshow(heatmap)
    axes[2].set_title('Isı Haritası')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path is None:
        filename = os.path.basename(image_path).replace('.jpg', '_gradcam.png')
        save_path = os.path.join(VISUALIZATIONS_DIR, filename)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"GRAD-CAM kaydedildi: {save_path}")
    return overlay, heatmap


def preprocess_for_gradcam(image: np.ndarray, size: tuple = (640, 640)) -> torch.Tensor:
    """Görüntüyü GRAD-CAM için hazırla."""
    # Yeniden boyutlandır
    image_resized = cv2.resize(image, size)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Tensor'a dönüştür (B, C, H, W)
    tensor = torch.from_numpy(image_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return tensor


def batch_gradcam(model, image_paths: list, target_layer, output_dir: str = None):
    """
    Birden fazla görüntü için GRAD-CAM oluştur.
    
    Args:
        model: PyTorch modeli
        image_paths: Görüntü yolları listesi
        target_layer: Hedef katman
        output_dir: Çıktı dizini
    """
    if output_dir is None:
        output_dir = VISUALIZATIONS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    for image_path in image_paths:
        try:
            filename = os.path.basename(image_path).replace('.jpg', '_gradcam.png')
            save_path = os.path.join(output_dir, filename)
            generate_gradcam_visualization(model, image_path, target_layer, save_path=save_path)
        except Exception as e:
            print(f"Hata ({image_path}): {e}")


def visualize_class_samples(model, image_paths: list, labels: list, target_layer,
                            samples_per_class: int = 1, output_dir: str = None):
    """
    Her sınıftan örnek GRAD-CAM görselleştirmesi.
    
    Args:
        model: PyTorch modeli
        image_paths: Görüntü yolları
        labels: Sınıf etiketleri
        target_layer: Hedef katman
        samples_per_class: Sınıf başına örnek sayısı
        output_dir: Çıktı dizini
    """
    if output_dir is None:
        output_dir = os.path.join(VISUALIZATIONS_DIR, 'class_samples')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Her sınıftan örnek seç
    unique_classes = np.unique(labels)
    
    for cls in unique_classes:
        cls_indices = np.where(np.array(labels) == cls)[0]
        selected = cls_indices[:samples_per_class]
        
        for idx in selected:
            image_path = image_paths[idx]
            class_name = CLASS_NAMES.get(cls, f'Class_{cls}')
            filename = f"{class_name.replace(' ', '_')}_{os.path.basename(image_path).replace('.jpg', '_gradcam.png')}"
            save_path = os.path.join(output_dir, filename)
            
            try:
                generate_gradcam_visualization(model, image_path, target_layer, 
                                               target_class=cls, save_path=save_path)
            except Exception as e:
                print(f"Hata ({class_name}): {e}")
