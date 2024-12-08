
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random


X = np.random.rand(1000, 1)
Y = 3 * X + 5 + np.random.normal(0, 0.1, (1000, 1))  # Doğrusal ilişki + gürültü

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)


hidden_layers = [1, 2, 3]          
neurons = [8, 16, 32, 64]         
learning_rates = [0.01, 0.001, 0.0001]  

best_val_loss = float('inf')  
best_params = None            


for _ in range(10):
    hl = random.choice(hidden_layers) 
    neuron = random.choice(neurons)    
    lr = random.choice(learning_rates) 

    print(f"Testing hidden_layers={hl}, neurons={neuron}, learning_rate={lr}")

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(1,)))  
    for _ in range(hl): 
        model.add(tf.keras.layers.Dense(neuron, activation='relu')) 
    model.add(tf.keras.layers.Dense(1))  

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse')  

  
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
              batch_size=32, epochs=50, verbose=0)

    val_loss = model.evaluate(X_val, Y_val, verbose=0)
    print(f"Validation Loss: {val_loss}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (hl, neuron, lr)

print(f"Best Hidden Layers: {best_params[0]}, Best Neurons: {best_params[1]}, "
      f"Best Learning Rate: {best_params[2]}, Best Validation Loss: {best_val_loss}")
