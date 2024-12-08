from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np

# BUrda yapay derin öğrenme verisi oluşturuyoruz
X, Y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
X = tf.convert_to_tensor(X, dtype=tf.float32)
Y = tf.convert_to_tensor(Y, dtype=tf.int32)

# Her foldda olacak modeli tanımlyıruz 
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(20,)),
        #İlk katman, 16 nöron ve ReLU aktivasyon fonksiyonunu kullanır. Bu katman, girdiden 16 özellik çıkarır.
        tf.keras.layers.Dense(8, activation='relu'),
        #İkinci katman, 8 nöron ve ReLU aktivasyonu ile çalışır.
        tf.keras.layers.Dense(1, activation='sigmoid')
        #Çıkış katmanı, ikili sınıflandırma için sigmoid aktivasyon fonksiyonunu kullanır.
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# K-Fold Çapraz Doğrulama
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#veriyi 5 e böldü ve flodlara karıştırdı    
fold_no = 1
accuracies = []

for train_idx, val_idx in kfold.split(X, Y):
    print(f"Fold {fold_no}")
    
    X_train, X_val = tf.gather(X, train_idx), tf.gather(X, val_idx)
    Y_train, Y_val = tf.gather(Y, train_idx), tf.gather(Y, val_idx)
    

    model = create_model()
    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)
    
   
    Y_pred = (model.predict(X_val) > 0.5).astype("int32")
    accuracy = accuracy_score(Y_val, Y_pred)
    accuracies.append(accuracy)
    print(f"Fold {fold_no} Accuracy: {accuracy}")
    
    fold_no += 1


print(f"Ortalama Accuracy: {np.mean(accuracies)}")
print(f"Accuracy Standart Sapması: {np.std(accuracies)}")

