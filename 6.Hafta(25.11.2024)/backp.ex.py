import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 100)
Y = 2 * X + 1
X_train = tf.convert_to_tensor(X, dtype=tf.float32)
Y_train = tf.convert_to_tensor(Y, dtype=tf.float32)

weights = tf.Variable(np.random.randn(), dtype=tf.float32)
bias = tf.Variable(np.random.randn(), dtype=tf.float32)

def model(X):
    return X * weights + bias

def compute_loss(Y_pred, Y_true):
    return tf.reduce_mean(tf.square(Y_pred - Y_true))

def train_step(X_batch, Y_batch):
    with tf.GradientTape() as tape:
        Y_pred = model(X_batch)
        loss = compute_loss(Y_pred, Y_batch)
    gradients = tape.gradient(loss, [weights, bias])
    learning_rate = 0.01
    weights.assign_sub(learning_rate * gradients[0])
    bias.assign_sub(learning_rate * gradients[1])
    return loss

epochs = 2000
for epoch in range(epochs):
    loss = train_step(X_train, Y_train)
    if epoch % 200 == 0:
        print(f"Epoch {epoch}: Loss = {loss.numpy()}")

Y_pred = model(X_train)

plt.scatter(X_train, Y_train, label='Gerçek Veriler')
plt.plot(X_train, Y_pred, color='red', label='Model Tahminleri')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()