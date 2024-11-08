from sklearn.linear_model import Lasso
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

# Lasso da yine ridge gibi amacı overfittingi önlemek bu sefer L1 regularizasyonunu kullanıyor. 
# ikisinin amacı aynı olsada kullandığı teknikler farklı ama bunu çok ayrıntı olacağını düşünüyorum 
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

y_pred_lasso = lasso_model.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)

plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='black', label='Gerçek Fiyatlar')
plt.plot(X_test, y_pred_lasso, color='green', label='Lasso Regression', linewidth=2)
plt.title(f'Lasso Regression\nMSE: {mse_lasso:.2f}')
plt.xlabel('Ev Büyüklüğü (sqft)')
plt.ylabel('Fiyat (USD)')
plt.legend()
plt.grid(True)
plt.show()
