# Parazit Yumurtası Türlerinin Füzyon Model ile Sınıflandırılması

**Classification of Parasite Egg Types with a Fusion Model**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)](https://ultralytics.com)

Bu çalışma, bağırsak paraziti yumurtalarının mikroskop görüntülerinden otomatik tespiti ve sınıflandırılması için derin öğrenme ve makine öğrenmesi yöntemlerinin füzyonunu içermektedir. Chula-ParasiteEgg-11 veri kümesi üzerinde gerçekleştirilen deneyler sonucunda, **F1 skoru %99.53** ve **mAP %95.62** başarımına ulaşılmıştır.

---

## İçindekiler

- [Veri Kümesi](#veri-kümesi)
- [Metodoloji](#metodoloji)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sonuçlar](#sonuçlar)
- [Proje Yapısı](#proje-yapısı)
- [Referanslar](#referanslar)

---
---

## Veri Kümesi

**Chula-ParasiteEgg-11** veri kümesi, IEEE ICIP 2022 yarışması kapsamında sunulmuştur.

| Sınıf ID | Parazit Türü | Eğitim | Test |
|:--------:|--------------|:------:|:----:|
| 0 | Ascaris lumbricoides | 1000 | 200 |
| 1 | Capillaria philippinensis | 1000 | 200 |
| 2 | Enterobius vermicularis | 1000 | 200 |
| 3 | Fasciolopsis buski | 1000 | 200 |
| 4 | Hookworm egg | 1000 | 200 |
| 5 | Hymenolepis diminuta | 1000 | 200 |
| 6 | Hymenolepis nana | 1000 | 200 |
| 7 | Opisthorchis viverrine | 1000 | 200 |
| 8 | Paragonimus spp | 1000 | 200 |
| 9 | Taenia spp. egg | 1000 | 200 |
| 10 | Trichuris trichiura | 1000 | 200 |

**Toplam**: 11.000 eğitim + 2.200 test görüntüsü

### Veri Ön İşleme

1. **HSV Dönüşümü**: RGB görüntüler HSV uzayına dönüştürülür
2. **Gaussian Bulanıklaştırma**: V kanalına gürültü azaltma
3. **CLAHE**: Kontrast iyileştirme (Contrast Limited Adaptive Histogram Equalization)
4. **Yeniden Boyutlandırma**: 1200x1200 piksel

### Veri Artırma (Augmentation)

| Veri Kümesi | Uygulanan İşlemler |
|-------------|-------------------|
| VeriKümesi#1 | Orijinal (sadece ön işleme) |
| VeriKümesi#2 | CLAHE + Tuz-Biber Gürültüsü + Sis Efekti |
| VeriKümesi#3 | Tuz-Biber + Sis + CLAHE + Yatay/Dikey Çevirme |

---

## Metodoloji

### Model Mimarisi

```
┌─────────────────────────────────────────────────────────────┐
│                    FÜZYON MODEL MİMARİSİ                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Görüntü   │───▶│  ViT-L/16   │───▶│  FPN/AFPN   │     │
│  │   Girişi    │    │  Backbone   │    │    Neck     │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │             │
│                                               ▼             │
│                                        ┌─────────────┐     │
│                                        │  YOLO11m    │     │
│                                        │    Head     │     │
│                                        └──────┬──────┘     │
│                                               │             │
│                                               ▼             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Öznitelik  │───▶│   XGBoost   │───▶│    Geç      │     │
│  │   Çıkarma   │    │  /RF/SVM    │    │   Füzyon    │     │
│  └─────────────┘    └─────────────┘    └──────┬──────┘     │
│                                               │             │
│                                               ▼             │
│                                        ┌─────────────┐     │
│                                        │   Tahmin    │     │
│                                        └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Derin Öğrenme Bileşenleri

| Bileşen | Açıklama |
|---------|----------|
| **Backbone** | Vision Transformer (ViT-L/16) |
| **Neck** | FPN (Feature Pyramid Network) veya AFPN (Asymptotic FPN) |
| **Head** | YOLO11 (s/m/x varyantları) |
| **C2f Modülü** | Gelişmiş özellik birleştirme |

### Makine Öğrenmesi Bileşenleri

Görüntülerden çıkarılan öznitelikler:

| Öznitelik | Açıklama |
|-----------|----------|
| area | Nesnenin kapladığı piksel sayısı |
| mean_intensity | Piksel yoğunluklarının ortalaması |
| LBP (12 öznitelik) | Local Binary Pattern - doku tanımı |
| LTP (12 öznitelik) | Local Ternary Pattern - hassas doku tanımı |
| RGB mean | Renk kanalları ortalaması |
| HSV mean | Ton, doygunluk, parlaklık ortalaması |
| spectral_entropy | Frekans karmaşıklığı |
| perimeter | Nesne çevresi |

### Füzyon Stratejisi

**Geç Füzyon (Late Fusion)**: Her iki modelin olasılık çıktıları ağırlıklı ortalama ile birleştirilir.

```
P_final = α × P_DL + (1-α) × P_ML
```

---

## Kurulum

### Gereksinimler

- Python 3.8+
- CUDA 11.0+ (GPU kullanımı için)

### Kurulum Adımları

```bash
# Repository'yi klonla
git clone <repo-url>
cd parasite-egg-classification

# Sanal ortam oluştur
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Bağımlılıkları yükle
pip install -r requirements.txt
```

---

## Kullanım

### 1. Veri Hazırlama

```bash
python main.py prepare
```

Bu komut:
- Görüntüleri ön işlemden geçirir
- YOLO formatında etiketler oluşturur
- Train/Val/Test split yapar
- `data.yaml` dosyasını oluşturur

### 2. Derin Öğrenme Eğitimi

```bash
# Varsayılan: YOLO11m + ViT + FPN, 300 epoch
python main.py train_dl

# Özelleştirilmiş eğitim
python main.py train_dl --model_type yolo11m --backbone vit --neck fpn --epochs 300

# Farklı model varyantları
python main.py train_dl --model_type yolo11s --epochs 20   # Hızlı test
python main.py train_dl --model_type yolo11x --epochs 200  # Maksimum performans
```

### 3. Makine Öğrenmesi Eğitimi

```bash
# Varsayılan parametrelerle
python main.py train_ml

# Grid Search ile hiperparametre optimizasyonu
python main.py train_ml --grid_search
```

### 4. Tüm Modelleri Eğit (Füzyon için)

```bash
python main.py train_all --model_type yolo11m --backbone vit --neck fpn --epochs 300
```

### 5. Model Test

```bash
python main.py test --model_path outputs/models/yolo11m_vit_fpn_e300.pt
```

### 6. Füzyon Tahmin

```bash
# XGBoost ile füzyon
python main.py fusion --ml_model xgboost --dl_weight 0.5

# RandomForest ile füzyon
python main.py fusion --ml_model random_forest --dl_weight 0.6
```

### 7. GRAD-CAM Görselleştirme

```bash
python main.py gradcam --model_path outputs/models/yolo11m_vit_fpn_e300.pt --image_path data/test/data/0001.jpg
```

---

## Sonuçlar

### Derin Öğrenme Model Karşılaştırması

| Model | Epoch | Doğruluk | Kesinlik | Hassasiyet | F1 Skor | mAP |
|-------|:-----:|:--------:|:--------:|:----------:|:-------:|:---:|
| ResNet50 | 30 | %49 | %57 | %50 | %49 | - |
| EfficientNetB4 | 30 | %54 | %59 | %53 | %55 | - |
| DenseNet121 | 30 | %59 | %68 | %58 | %59 | - |
| ViT-L/16 | 30 | %64 | %69 | %63 | %66 | - |
| YOLO11s | 20 | %87.54 | %88.40 | %87.15 | %87.78 | %85.31 |
| YOLO11m | 20 | %88.73 | %89.12 | %88.23 | %88.87 | %86.01 |
| YOLO11m-ViT-AFPN | 300 | %98.52 | %98.19 | %97.81 | %98.02 | %94.42 |
| **YOLO11m-ViT-FPN** | **300** | **%99.73** | **%99.81** | **%99.13** | **%99.53** | **%95.62** |

### Makine Öğrenmesi Model Karşılaştırması

| Model | Doğruluk | Kesinlik | Hassasiyet | F1 Skor |
|-------|:--------:|:--------:|:----------:|:-------:|
| Logistic Regression | %62.36 | %65.18 | %62.82 | %64.15 |
| KNN | %71.13 | %72.14 | %69.81 | %70.76 |
| RandomForest | %79.02 | %79.49 | %78.12 | %78.51 |
| **XGBoost** | **%80.12** | **%82.12** | **%80.42** | **%81.74** |

### Literatür Karşılaştırması

| Yıl | Referans | Model | F1 Skor | mAP |
|:---:|----------|-------|:-------:|:---:|
| 2024 | Xu et al. | YAC-Net | %97.73 | - |
| 2023 | Rajasekar et al. | YOLOv8 | %98 | %92 |
| 2023 | AlDahoul et al. | CoAtNet | %94 | - |
| 2022 | Tureckova et al. | TOOD-r101 | %96.46 | - |
| **2025** | **Bu Çalışma** | **YOLO11m-ViT-FPN** | **%99.53** | **%95.62** |

---

## Proje Yapısı

```
parasite-egg-classification/
│
├── main.py                # Ana çalıştırma scripti (CLI)
├── config.py              # Konfigürasyon ve hiperparametreler
├── requirements.txt       # Python bağımlılıkları
├── README.md              # Proje dokümantasyonu
├── LICENSE
├── .gitignore
│
├── src/                   # Kaynak kodlar
│   ├── __init__.py
│   ├── dataset.py         # Veri hazırlama ve YOLO formatı
│   ├── preprocessing.py   # Görüntü ön işleme fonksiyonları
│   ├── features.py        # Öznitelik çıkarma (LBP, LTP, vb.)
│   ├── models.py          # Derin öğrenme modelleri (ViT, FPN, AFPN)
│   ├── ml_models.py       # Makine öğrenmesi modelleri
│   ├── train.py           # Eğitim scriptleri
│   ├── evaluate.py        # Değerlendirme ve metrikler
│   ├── fusion.py          # Model füzyonu
│   ├── gradcam.py         # GRAD-CAM görselleştirme
│   └── utils.py           # Yardımcı fonksiyonlar
│
├── data/
│   ├── train/
│   │   ├── data/          # Eğitim görselleri (11.000 adet)
│   │   └── labels.json    # Eğitim etiketleri
│   ├── test/
│   │   └── data/          # Test görselleri (2.200 adet)
│   └── test_labels_200.json
│
└── outputs/               # Çalıştırma sonrası oluşur
    ├── models/            # Eğitilmiş modeller
    ├── predictions/       # Tahmin sonuçları
    └── visualizations/    # GRAD-CAM, karışıklık matrisi vb.
```

---

## Referanslar

```bibtex
@inproceedings{anantrasirichai2022icip,
  title={ICIP 2022 Challenge on Parasitic Egg Detection and Classification in Microscopic Images},
  author={Anantrasirichai, N. and others},
  booktitle={IEEE International Conference on Image Processing (ICIP)},
  year={2022}
}
```

