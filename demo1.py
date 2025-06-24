# EEG Emotion Recognition Demo Notebook

# Cell 1: Import ve Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set_palette("husl")

print("✅ Kütüphaneler yüklendi!")

# Cell 2: Sentetik EEG Verisi Oluşturma
def generate_eeg_sample(emotion, n_channels=14, signal_length=1024, fs=128):
    """Belirli bir duygu için EEG sinyali oluştur"""
    
    # Duygu bazlı frekans profilleri
    profiles = {
        0: {'alpha': 2.0, 'beta': 0.5, 'theta': 1.0},  # sakin
        1: {'alpha': 1.2, 'beta': 2.0, 'theta': 0.8},  # mutlu
        2: {'alpha': 0.6, 'beta': 0.7, 'theta': 1.8},  # üzgün
        3: {'alpha': 0.8, 'beta': 1.5, 'theta': 1.2}   # öfkeli
    }
    
    profile = profiles[emotion]
    t = np.linspace(0, signal_length/fs, signal_length)
    
    eeg_data = np.zeros((n_channels, signal_length))
    
    for ch in range(n_channels):
        # Farklı frekans bantları
        delta = np.sin(2*np.pi*2*t) * np.random.uniform(0.5, 1.0)
        theta = np.sin(2*np.pi*6*t) * profile['theta'] * np.random.uniform(0.8, 1.2)
        alpha = np.sin(2*np.pi*10*t) * profile['alpha'] * np.random.uniform(0.8, 1.2)
        beta = np.sin(2*np.pi*20*t) * profile['beta'] * np.random.uniform(0.8, 1.2)
        
        # Gürültü
        noise = np.random.normal(0, 0.1, signal_length)
        
        eeg_data[ch, :] = delta + theta + alpha + beta + noise
    
    return eeg_data

# Örnek sinyaller oluştur
emotion_names = ['Sakin', 'Mutlu', 'Üzgün', 'Öfkeli']
sample_signals = {}

for i, emotion in enumerate(emotion_names):
    sample_signals[emotion] = generate_eeg_sample(i)

print("✅ Örnek EEG sinyalleri oluşturuldu!")

# Cell 3: Sinyal Görselleştirme
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, (emotion, signal_data) in enumerate(sample_signals.items()):
    # İlk 3 kanalı göster
    t = np.linspace(0, 8, 1024)  # 8 saniye
    
    for ch in range(3):
        axes[i].plot(t, signal_data[ch, :] + ch*2, label=f'Kanal {ch+1}')
    
    axes[i].set_title(f'{emotion} - EEG Sinyali')
    axes[i].set_xlabel('Zaman (s)')
    axes[i].set_ylabel('Amplitüd (µV)')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 4: Güç Spektrumu Analizi
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, (emotion, signal_data) in enumerate(sample_signals.items()):
    # İlk kanalın güç spektrumunu hesapla
    freqs, psd = signal.welch(signal_data[0, :], 128, nperseg=128)
    
    axes[i].plot(freqs[:50], psd[:50])  # 0-50 Hz arası
    axes[i].set_title(f'{emotion} - Güç Spektrumu')
    axes[i].set_xlabel('Frekans (Hz)')
    axes[i].set_ylabel('Güç (µV²/Hz)')
    axes[i].grid(True, alpha=0.3)
    
    # Frekans bantlarını vurgula
    axes[i].axvspan(0.5, 4, alpha=0.2, color='red', label='Delta')
    axes[i].axvspan(4, 8, alpha=0.2, color='orange', label='Theta')
    axes[i].axvspan(8, 13, alpha=0.2, color='green', label='Alpha')
    axes[i].axvspan(13, 30, alpha=0.2, color='blue', label='Beta')
    
    if i == 0:
        axes[i].legend()

plt.tight_layout()
plt.show()

# Cell 5: Özellik Çıkarımı
def extract_band_powers(eeg_data, fs=128):
    """Frekans bantı güçlerini çıkar"""
    features = []
    
    for ch in range(eeg_data.shape[0]):
        freqs, psd = signal.welch(eeg_data[ch, :], fs, nperseg=128)
        
        # Frekans bantları
        delta = np.mean(psd[(freqs >= 0.5) & (freqs <= 4)])
        theta = np.mean(psd[(freqs >= 4) & (freqs <= 8)])
        alpha = np.mean(psd[(freqs >= 8) & (freqs <= 13)])
        beta = np.mean(psd[(freqs >= 13) & (freqs <= 30)])
        
        # Relatif güçler
        total = delta + theta + alpha + beta
        
        features.extend([
            delta/total, theta/total, alpha/total, beta/total,
            np.mean(eeg_data[ch, :]), np.std(eeg_data[ch, :])
        ])
    
    return np.array(features)

# Veri seti oluştur
print("Veri seti oluşturuluyor...")
X, y = [], []

for emotion_idx in range(4):
    for _ in range(250):  # Her duygu için 250 örnek
        eeg_signal = generate_eeg_sample(emotion_idx)
        features = extract_band_powers(eeg_signal)
        X.append(features)
        y.append(emotion_idx)

X = np.array(X)
y = np.array(y)

print(f"✅ Veri seti hazır: {X.shape[0]} örnek, {X.shape[1]} özellik")
print(f"Sınıf dağılımı: {np.bincount(y)}")

# Cell 6: Model Eğitimi
# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model eğitimi
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Tahmin
y_pred = rf_model.predict(X_test_scaled)
accuracy = np.mean(y_pred == y_test)

print(f"✅ Model eğitildi!")
print(f"Test Doğruluğu: %{accuracy*100:.2f}")

# Cell 7: Sonuçları Görselleştir
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=emotion_names, yticklabels=emotion_names, ax=axes[0,0])
axes[0,0].set_title('Confusion Matrix')

# Feature Importance
importances = rf_model.feature_importances_
feature_names = []
for ch in range(14):
    for feat in ['Delta', 'Theta', 'Alpha', 'Beta', 'Mean', 'Std']:
        feature_names.append(f'Ch{ch+1}_{feat}')

# Top 20 özellik
top_indices = np.argsort(importances)[-20:]
axes[0,1].barh(range(20), importances[top_indices])
axes[0,1].set_yticks(range(20))
axes[0,1].set_yticklabels([feature_names[i] for i in top_indices])
axes[0,1].set_title('Top 20 Önemli Özellikler')

# Sınıf bazında doğruluk
from sklearn.metrics import precision_recall_fscore_support
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)

x_pos = np.arange(len(emotion_names))
width = 0.25

axes[1,0].bar(x_pos - width, precision, width, label='Precision', alpha=0.8)
axes[1,0].bar(x_pos, recall, width, label='Recall', alpha=0.8)
axes[1,0].bar(x_pos + width, f1, width, label='F1-Score', alpha=0.8)

axes[1,0].set_xlabel('Duygu')
axes[1,0].set_ylabel('Skor')
axes[1,0].set_title('Sınıf Bazında Performans')
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(emotion_names)
axes[1,0].legend()

# Örnek tahmin olasılıkları
sample_probs = rf_model.predict_proba(X_test_scaled[:10])
im = axes[1,1].imshow(sample_probs.T, cmap='YlOrRd', aspect='auto')
axes[1,1].set_title('İlk 10 Test Örneği - Tahmin Olasılıkları')
axes[1,1].set_xlabel('Test Örneği')
axes[1,1].set_ylabel('Duygu Sınıfı')
axes[1,1].set_yticks(range(4))
axes[1,1].set_yticklabels(emotion_names)
plt.colorbar(im, ax=axes[1,1])

plt.tight_layout()
plt.show()

# Cell 8: Detaylı Rapor
print("=" * 50)
print("DETAYLI PERFORMANS RAPORU")
print("=" * 50)

print(classification_report(y_test, y_pred, target_names=emotion_names))

# Cell 9: Gerçek Zamanlı Tahmin Simulasyonu
print("\n" + "="*50)
print("GERÇEK ZAMANLI TAHMİN SİMÜLASYONU")
print("="*50)

# Yeni bir örnek oluştur ve tahmin et
for i in range(5):
    # Rastgele bir duygu seç
    true_emotion = np.random.randint(0, 4)
    
    # Bu duygu için EEG sinyali oluştur
    test_signal = generate_eeg_sample(true_emotion)
    test_features = extract_band_powers(test_signal).reshape(1, -1)
    test_features_scaled = scaler.transform(test_features)
    
    # Tahmin yap
    prediction = rf_model.predict(test_features_scaled)[0]
    probabilities = rf_model.predict_proba(test_features_scaled)[0]
    
    print(f"\nÖrnek {i+1}:")
    print(f"Gerçek Duygu: {emotion_names[true_emotion]}")
    print(f"Tahmin: {emotion_names[prediction]} {'✅' if prediction == true_emotion else '❌'}")
    print("Olasılıklar:")
    for j, prob in enumerate(probabilities):
        print(f"  {emotion_names[j]}: %{prob*100:.1f}")

# Cell 10: Model Kaydetme
import pickle

# Modeli kaydet
model_data = {
    'model': rf_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'emotion_names': emotion_names
}

with open('eeg_emotion_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model kaydedildi: eeg_emotion_model.pkl")

# Özet
print("\n" + "="*60)
print("PROJE ÖZETİ")
print("="*60)
print(f"📊 Veri: {X.shape[0]} örnek, {X.shape[1]} özellik")
print(f"🎯 Sınıf: {len(emotion_names)} duygu durumu")
print(f"🤖 Model: Random Forest Classifier")
print(f"📈 Doğruluk: %{accuracy*100:.2f}")
print(f"💾 Model dosyası: eeg_emotion_model.pkl")
print("\n🚀 Proje hazır!  sunuma gitmeye hazırsınız!")