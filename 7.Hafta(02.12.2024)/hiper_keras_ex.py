import tensorflow as tf
from tensorflow import keras
import keras
from keras import layers
import keras_tuner as kt
import tempfile

# Veri seti yükleme ve işleme (örnek için MNIST)
(X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28 * 28)
X_test = X_test.reshape(-1, 28 * 28)

# Veri setini eğitim ve doğrulama olarak ayırma
X_val, Y_val = X_train[-10000:], Y_train[-10000:]
X_train, Y_train = X_train[:-10000], Y_train[:-10000]

# Model oluşturma fonksiyonu
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Input(shape=(28 * 28,)))
    
    # Hiperparametrelere göre gizli katmanlar eklenir
    for i in range(hp.Int("hidden_layers", 1, 3)):  # 1-3 arasında gizli katman sayısı
        model.add(layers.Dense(units=hp.Choice("units", [16, 32, 64]), activation="relu"))
    
    # Çıkış katmanı
    model.add(layers.Dense(10, activation="softmax"))
    
    # Öğrenme oranı hiperparametresine göre optimizasyon fonksiyonu
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice("learning_rate", [0.01, 0.001, 0.0001])),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# Geçici dizin oluşturma
temp_dir = tempfile.mkdtemp()

# Keras Tuner ile hiperparametre araması
tuner = kt.RandomSearch(
    build_model,
    objective="val_loss",
    max_trials=10,
    executions_per_trial=2,
    directory=temp_dir,  # Geçici dizin kullanımı
    project_name="temp_project",
)

# Hiperparametre optimizasyonu
tuner.search(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), verbose=1)

# En iyi hiperparametreleri seçme
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("En iyi hiperparametreler:")
print(best_hps.values)

# En iyi modelin yeniden eğitilmesi
model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, Y_train, epochs=10, validation_data=(X_val, Y_val), verbose=1)

# Test verisinde değerlendirme
test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
print(f"Test doğruluğu: {test_acc}")
