# Sensoria - EEG Tabanlı Duygu Tanıma Sistemi

Bu proje, EEG (Elektroensefalogram) sinyallerinden makine öğrenmesi tekniklerini kullanarak duygu tanıma yapmaktadır.

## 🎯 Proje Hedefi

EEG sinyallerinden 4 temel duygu durumunu sınıflandırma:
- **Sakin** (Calm)
- **Mutlu** (Happy) 
- **Üzgün** (Sad)
- **Öfkeli** (Angry)

## 📊 Metodoloji

### Özellik Çıkarımı
- **İstatistiksel Özellikler**: Ortalama, standart sapma, varyans, çarpıklık, basıklık
- **Frekans Domain Özellikleri**: 
  - Delta (0.5-4 Hz)
  - Theta (4-8 Hz) 
  - Alpha (8-13 Hz)
  - Beta (13-30 Hz)
  - Gamma (30-50 Hz)
- **Spektral Özellikler**: Güç spektral yoğunluğu analizi

### Model Pipeline
1. **Ön İşleme**: StandardScaler ile normalizasyon
2. **Boyut Azaltma**: PCA (50 bileşen)
3. **Sınıflandırma**: Random Forest Classifier
4. **Değerlendirme**: Confusion matrix, F1-score, accuracy

## 🚀 Kurulum

```bash
# Repository'yi klonlayın
git clone https://github.com/barisparlakk/Sensoria.git
cd Sensoria

# Gerekli paketleri yükleyin
pip install -r Requirements.txt

# Ana scripti çalıştırın
python main.py
```

## 📈 Sonuçlar

Mevcut model sentetik veri üzerinde **~xx-xx%** doğruluk oranı elde etmektedir.

### Model Performansı
- **Accuracy**: ~xx%
- **Precision**: Sınıf bazında xx-xx
- **Recall**: Sınıf bazında xx-xx
- **F1-Score**: Sınıf bazında xx-xx

## 📁 Dosya Yapısı

```
eeg-emotion-recognition/
├── main.py                 # Ana script
├── requirements.txt        # Python paketleri
├── README.md              # Bu dosya
├── eeg_emotion_model.pkl   # Eğitilmiş model (çalıştırma sonrası)
└── eeg_emotion_results.png # Sonuç grafikleri (çalıştırma sonrası)
```

## 🔬 Teknik Detaylar

### Veri İşleme
- **Sampling Rate**: 128 Hz
- **Kanal Sayısı**: 14 kanal (sentetik veri)
- **Sinyal Uzunluğu**: 1024 sample (~8 saniye)

### Feature Engineering
- Kanal başına 16 özellik çıkarılır
- Toplam 224 özellik (14 kanal × 16 özellik)
- PCA ile 50 ana bileşene indirgenir

### Model Hiperparametreleri
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
```

## 🔄 Gerçek Veri Kullanımı

Gerçek projede aşağıdaki datasetler kullanılabilir:

### DEAP Dataset
```python
# DEAP dataset yükleme örneği
import pickle
data = pickle.load(open('s01.dat', 'rb'), encoding='latin1')
eeg_data = data['data']  # (40 trials, 40 channels, 8064 samples)
labels = data['labels']  # (40 trials, 4 values: valence, arousal, dominance, liking)
```

### Veri Ön İşleme
```python
# Filtreleme ve artefakt temizleme
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.5, highcut=50, fs=128):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    return filtfilt(b, a, data)
```

## 📊 Görselleştirme

Script aşağıdaki grafikleri oluşturur:
1. **Confusion Matrix**: Sınıflandırma performansı
2. **Feature Importance**: En önemli özellikler
3. **PCA Visualization**: 2D dağılım grafiği
4. **F1-Score by Class**: Sınıf bazında performans

## 🎓 Akademik Referanslar

- Koelstra, S., et al. (2012). DEAP: A database for emotion analysis using physiological signals.
- Russell, J. A. (1980). A circumplex model of affect.
- Davidson, R. J. (2004). What does the prefrontal cortex "do" in affect?

## 🚧 Gelecek Geliştirmeler

- [ ] Gerçek EEG dataset entegrasyonu (DEAP, SEED)
- [ ] Deep Learning modelleri (CNN, LSTM)
- [ ] Real-time EEG sinyal işleme
- [ ] Web arayüzü geliştirme
- [ ] Artefakt temizleme algoritmaları
- [ ] Cross-subject validation

## 👥 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Commit yapın (`git commit -am 'Yeni özellik eklendi'`)
4. Push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📧 İletişim

Sorularınız için: [barisparlak36@gmail.com]

---

**Not**: Bu proje eğitim amaçlı hazırlanmıştır. Gerçek uygulamalarda profesyonel EEG cihazları ve validated datasetler kullanılmalıdır.