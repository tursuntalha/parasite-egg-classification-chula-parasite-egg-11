# -*- coding: utf-8 -*-
"""
Derin öğrenme modelleri - YOLO, ViT, FPN, AFPN
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO


class TransformerBackbone(nn.Module):
    """Vision Transformer (ViT-L/16) Backbone."""
    
    def __init__(self, pretrained: bool = True):
        super(TransformerBackbone, self).__init__()
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.vit = models.vision_transformer.vit_l_16(weights=weights)
        self.vit = nn.Sequential(*list(self.vit.children())[:-2])

    def forward(self, x):
        return self.vit(x)


class FPN(nn.Module):
    """Feature Pyramid Network."""
    
    def __init__(self, in_channels: list, out_channels: int):
        super(FPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        self.fpn = nn.ModuleList([self.conv1, self.conv2, self.conv3])

    def forward(self, x):
        features = [conv(x[i]) for i, conv in enumerate(self.fpn)]
        return features


class AFPN(nn.Module):
    """Asymptotic Feature Pyramid Network."""
    
    def __init__(self, in_channels: list, out_channels: int):
        super(AFPN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(in_channels[1], out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels[2], out_channels, kernel_size=1)
        
        # Asimptotik özellik birleştirme katmanları
        self.asymp_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        features = [self.conv1(x[0]), self.conv2(x[1]), self.conv3(x[2])]
        
        # Asimptotik özellik birleştirme
        for i in range(len(features)):
            features[i] = self.asymp_conv(features[i])
            features[i] = self.bn(features[i])
            features[i] = self.activation(features[i])
        
        return features


class C2f(nn.Module):
    """C2f Modülü - YOLOv8/11 için."""
    
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False):
        super(C2f, self).__init__()
        self.c = int(c2)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, 1, 1)
        self.cv2 = nn.Conv2d((2 + n) * self.c, c2, 1)
        self.bn = nn.BatchNorm2d(2 * self.c)
        self.act = nn.SiLU()
        self.m = nn.ModuleList([nn.Conv2d(self.c, self.c, 3, 1, 1) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class YOLOWithTransformerFPN(YOLO):
    """YOLO + ViT + FPN entegrasyonu."""
    
    def __init__(self, pretrained_model_path: str = 'yolo11m.pt'):
        super(YOLOWithTransformerFPN, self).__init__(pretrained_model_path)
        self.transformer_backbone = TransformerBackbone()
        self.fpn = FPN([1024, 1024, 1024], 256)

    def forward(self, x):
        transformer_features = self.transformer_backbone(x)
        fpn_features = self.fpn([transformer_features] * 3)
        return fpn_features


class YOLOWithTransformerAFPN(YOLO):
    """YOLO + ViT + AFPN + C2f entegrasyonu."""
    
    def __init__(self, pretrained_model_path: str = 'yolo11m.pt'):
        super(YOLOWithTransformerAFPN, self).__init__(pretrained_model_path)
        self.transformer_backbone = TransformerBackbone()
        self.afpn = AFPN([1024, 1024, 1024], 256)
        self.c2f = C2f(256, 256)

    def forward(self, x):
        transformer_features = self.transformer_backbone(x)
        afpn_features = self.afpn([transformer_features] * 3)
        enhanced_features = [self.c2f(feat) for feat in afpn_features]
        return enhanced_features


def get_model(model_type: str = 'yolo11m', backbone: str = None, 
              neck: str = None, pretrained_path: str = None):
    """
    Model seçim fonksiyonu.
    
    Args:
        model_type: 'yolo11s', 'yolo11m', 'yolo11x'
        backbone: None, 'vit'
        neck: None, 'fpn', 'afpn'
        pretrained_path: Önceden eğitilmiş model yolu
    """
    if pretrained_path is None:
        pretrained_path = f'{model_type}.pt'
    
    if backbone == 'vit' and neck == 'fpn':
        return YOLOWithTransformerFPN(pretrained_path)
    elif backbone == 'vit' and neck == 'afpn':
        return YOLOWithTransformerAFPN(pretrained_path)
    else:
        return YOLO(pretrained_path)


def get_base_yolo(model_size: str = 'm') -> YOLO:
    """Temel YOLO modeli döndür."""
    model_map = {
        's': 'yolo11s.pt',
        'm': 'yolo11m.pt',
        'x': 'yolo11x.pt'
    }
    return YOLO(model_map.get(model_size, 'yolo11m.pt'))
