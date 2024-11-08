import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {'House Size (m2)': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000],
        'Bedrooms': [3, 4, 3, 5, 4, 5, 6, 5],
        'Price': [400000, 450000, 500000, 600000, 650000, 700000, 750000, 800000]}

df = pd.DataFrame(data)

X = df[['House Size (m2)']]  #metrekareyi kullanıyoruz
y = df['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


lr_model = LinearRegression() #En yaygın regresyon yöntemidir. Hedef değişken, bağımsız değişkenlerin doğrusal bir kombinasyonu olarak modellenir
lr_model.fit(X_train, y_train)


y_pred_lr = lr_model.predict(X_test)

#buraya kadar hersey tamam simdi MSE gerçek değerden ne kadar saptığını ifade ediyor ne kadar küçükse o kadar iyi. 
mse_lr = mean_squared_error(y_test, y_pred_lr)

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='black', label='Gerçek Fiyatlar')
plt.plot(X_test, y_pred_lr, color='blue', label='Linear Regression', linewidth=2)
plt.title(f'Linear Regression\nMSE: {mse_lr:.2f}')
plt.xlabel('Ev Büyüklüğü (m2)')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.grid(True)
plt.show()
