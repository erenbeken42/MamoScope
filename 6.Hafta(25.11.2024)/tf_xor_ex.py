import numpy as np
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

# isminden de analşıalcağı üzere
# Modelimiz sıralı bi yapıda yani her katman bir önceki çıktıyı kullanarak ilerler

model = Sequential([
    Dense(8, input_dim=2, activation='relu'),  # Gizli katman 
    Dense(1, activation='sigmoid')            # sigmoid --> değeri 0 ila 1 arasına sıkıstırıyor
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=1000, verbose=0)#burda da epocha kısmı önemli önemini rapora yazacağım.

predictions = model.predict(X)
print("Tahminler:", predictions)
