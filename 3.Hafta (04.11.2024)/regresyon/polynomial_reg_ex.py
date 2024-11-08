from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


data = {'House Size (sqft)': [1500, 1800, 2400, 3000, 3500, 4000, 4500, 5000],
        'Bedrooms': [3, 4, 3, 5, 4, 5, 6, 5],
        'Price': [400000, 450000, 500000, 600000, 650000, 700000, 750000, 800000]}

df = pd.DataFrame(data)


X = df[['House Size (sqft)']]  
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)# x trainin içindeki house size ı polynoma çeviriyoruz yukarıdada hangi derecede çevireceğimiz yazıyor 

poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

X_test_poly = poly.transform(X_test) #Test verisini de aynı şekilde polinomal özelliklerle dönüştürüyoruz (poly.transform(X_test)). hmmmm
y_pred_poly = poly_model.predict(X_test_poly)
#Polinomal regresyon, doğrusal regresyon modelini genişleterek veriyi 
#polinomal bir şekle dönüştürür. Özellikle doğrusal olmayan ilişkiler için faydalıdır.
mse_poly = mean_squared_error(y_test, y_pred_poly)

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='black', label='Gerçek Fiyatlar')
plt.plot(X_test, y_pred_poly, color='orange', label='Polynomial Regression', linewidth=2)
plt.title(f'Polynomial Regression\nMSE: {mse_poly:.2f}')
plt.xlabel('Ev Büyüklüğü (sqft)')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.grid(True)
plt.show()
