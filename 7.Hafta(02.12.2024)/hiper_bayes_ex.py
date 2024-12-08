import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.base import BaseEstimator, RegressorMixin

# Veri oluşturma
X = np.random.rand(1000, 1)  # 1000 adet rastgele giriş verisi
Y = 3 * X + 5 + np.random.normal(0, 0.1, (1000, 1))  # Doğrusal ilişki + gürültü

# Eğitim ve doğrulama setine ayırma
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Modeli tanımlama fonksiyonu
def create_model(hidden_layers=2, neurons=32, learning_rate=0.001):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(1,)))  # Giriş katmanı
    for _ in range(hidden_layers):  # Gizli katman sayısı
        model.add(tf.keras.layers.Dense(neurons, activation='relu'))  # ReLU aktivasyonu ile katman
    model.add(tf.keras.layers.Dense(1))  # Çıkış katmanı (regresyon)
    
    # Modeli derleme
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')  # Kayıp fonksiyonu: MSE
    return model

# Keras modelini sklearn ile uyumlu hale getirme
class KerasRegressorCustom(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layers=2, neurons=32, learning_rate=0.001):
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y):
        self.model = create_model(self.hidden_layers, self.neurons, self.learning_rate)
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Hiperparametrelerin aralıklarını sözlük şeklinde tanımlama
param_space = {
    'hidden_layers': Integer(1, 3),  # Gizli katman sayısı
    'neurons': Integer(8, 64),       # Nöron sayısı
    'learning_rate': Real(1e-4, 1e-2, prior='log-uniform')  # Öğrenme oranı
}

# Keras modelini sklearn ile uyumlu hale getirme
opt = BayesSearchCV(KerasRegressorCustom(), param_space, n_iter=10, cv=3, n_jobs=-1)

# Optimizasyonu başlatma
opt.fit(X_train, Y_train)

# Sonuçları yazdırma
print("Best Hyperparameters: ", opt.best_params_)
print("Best Validation Loss: ", opt.best_score_)
