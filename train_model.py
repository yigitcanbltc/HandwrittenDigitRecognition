import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import os

# Proje klasörünün yolunu al
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Veri setlerinin tam yollarını oluştur
TRAIN_PATH = os.path.join(PROJECT_DIR, 'Datasets', 'train.csv')
TEST_PATH = os.path.join(PROJECT_DIR, 'Datasets', 'test.csv')
MODEL_PATH = os.path.join(PROJECT_DIR, 'handwritten_digit_model.keras')

# Eğitim ve test verilerini yükleme
try:
    print(f"Veri seti yükleniyor: {TRAIN_PATH}")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
except FileNotFoundError:
    print("Hata: Veri seti dosyaları bulunamadı!")
    print("Lütfen aşağıdaki dosyaların varlığını kontrol edin:")
    print(f"- {TRAIN_PATH}")
    print(f"- {TEST_PATH}")
    exit(1)

print("Veri seti başarıyla yüklendi. Eğitim başlıyor...")

# Etiketler (labels) ve veri (features) ayrılması
Y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

# Veriyi normalize etme
X_train = X_train / 255.0

# Veriyi şekillendirme
X_train = X_train.values.reshape(-1, 28, 28, 1)

# Etiketleri one-hot encoding formatına dönüştürme
Y_train = to_categorical(Y_train, num_classes=10)

# Eğitim ve doğrulama verilerini ayırma
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Modeli derleme
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

print("Model eğitimi başlıyor...")

# Modeli eğitme
history = model.fit(X_train, Y_train, 
                   epochs=10, 
                   batch_size=32, 
                   validation_data=(X_val, Y_val),
                   verbose=1)

print(f"Model kaydediliyor: {MODEL_PATH}")

# Modeli kaydet
model.save(MODEL_PATH)
print("Model başarıyla kaydedildi!") 