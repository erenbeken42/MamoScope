import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(0, 10, 100)
Y = 2 * X + 1 + np.random.normal(0, 1, 100)
X_train = tf.convert_to_tensor(X, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y, dtype=tf.float32)

class LinearModel(tf.Module):
    def __init__(self):
        self.w = tf.Variable(np.random.randn(), dtype=tf.float32)
        self.b = tf.Variable(np.random.randn(), dtype=tf.float32)
    
    def __call__(self, x):
        return self.w * x + self.b


def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Optimizasyon algoritmaları
optimizers = {
    "SGD": tf.keras.optimizers.SGD(learning_rate=0.01),
    "Momentum": tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSProp": tf.keras.optimizers.RMSprop(learning_rate=0.01),
    "Adam": tf.keras.optimizers.Adam(learning_rate=0.01),
}

# Eğitim döngüsü ve sonuçların saklanması
results = {}
for name, optimizer in optimizers.items():
    model = LinearModel()  # Her algoritma için yeni model oluştur
    losses = []
    for epoch in range(500):
        with tf.GradientTape() as tape:
            y_pred = model(X_train)
            loss = compute_loss(y_pred, Y_train)
        grads = tape.gradient(loss, [model.w, model.b])
        optimizer.apply_gradients(zip(grads, [model.w, model.b]))
        losses.append(loss.numpy())
    results[name] = (model.w.numpy(), model.b.numpy(), losses)
    print(f"{name}: w={model.w.numpy()}, b={model.b.numpy()}, Final Loss={loss.numpy()}")

# Kayıp değerlerini çizme
plt.figure(figsize=(12, 8))
for name, (_, _, losses) in results.items():
    plt.plot(losses, label=name)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Optimizasyon Tekniklerinin Karşılaştırması')
plt.show()
