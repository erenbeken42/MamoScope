import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Giriş Verisi (aralık: -10 ile 10 arası)
x = np.linspace(-10, 10, 100)

y_relu = relu(x)
y_sigmoid = sigmoid(x)

# Grafik
plt.figure(figsize=(10, 6))

# ReLU fonksiyonunun grafiği
plt.subplot(1, 2, 1)
plt.plot(x, y_relu, color='green')
plt.title("ReLU Fonksiyonu")
plt.xlabel("Girdi (x)")
plt.ylabel("Çıktı (ReLU(x))")
plt.grid()

# Sigmoid fonksiyonunun grafiği
plt.subplot(1, 2, 2)
plt.plot(x, y_sigmoid, color='blue')
plt.title("Sigmoid Fonksiyonu")
plt.xlabel("Girdi (x)")
plt.ylabel("Çıktı (Sigmoid(x))")
plt.grid()

plt.tight_layout()
plt.show()
