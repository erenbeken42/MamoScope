import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

X = np.random.rand(1000, 1)
Y = 3 * X + 5 + np.random.normal(0, 0.1, (1000, 1))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

learning_rates = [0.01, 0.001, 0.0001]
batch_sizes = [16, 32, 64]
best_val_loss = float('inf')
best_params = None

for lr in learning_rates:
    for batch_size in batch_sizes:
        print(f"Testing lr={lr}, batch_size={batch_size}")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss='mse')
        
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                            batch_size=batch_size, epochs=50, verbose=0)
        
        val_loss = model.evaluate(X_val, Y_val, verbose=0)
        print(f"Validation Loss: {val_loss}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = (lr, batch_size)

print(f"Best Learning Rate: {best_params[0]}, Best Batch Size: {best_params[1]}, Best Validation Loss: {best_val_loss}")
