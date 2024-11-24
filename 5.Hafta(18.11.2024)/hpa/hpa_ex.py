import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

train_data_cleaned = pd.read_csv('train_cleaned.csv')

test_data_cleaned = pd.read_csv('test_cleaned.csv')

#korelasyon sonucunda elde ettiğmiiz özellikler
important_features = [
    'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 
    '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 
    'GarageCars', 'GarageArea'
]

X_train = train_data_cleaned[important_features]
y_train = train_data_cleaned['SalePrice']

X_test = test_data_cleaned[important_features]

X_train = X_train.fillna(X_train.median())
X_test = X_test.fillna(X_test.median())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Submission dosyası için ID ve tahmin edilen SalePrice hazırlayalım
submission = pd.DataFrame({
    'Id': test_data_cleaned['Id'],  # Test verisindeki IDleri alıyoruz
    'SalePrice': y_pred  # Tahmin edilen SalePrice değerlerini alıyoruz
})

submission.to_csv('submission.csv', index=False)

print("Submission dosyası oluşturuldu: submission.csv")

y_train_pred = model.predict(X_train_scaled)

# Değerlendirme metrikleri
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)


print("\nEğitim Seti Değerlendirme Metrikleri:")
print(f"Mean Squared Error (MSE) - Eğitim Seti: {mse_train}")
print(f"Root Mean Squared Error (RMSE) - Eğitim Seti: {rmse_train}")
print(f"R-squared (R²) - Eğitim Seti: {r2_train}")
