import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train_data_cleaned = pd.read_csv('train_cleaned.csv')

numerical_columns = train_data_cleaned.select_dtypes(include=['float64', 'int64']).columns

# Sayısal özelliklerin korelasyon matrisini hesaplayalım
correlation_matrix = train_data_cleaned[numerical_columns].corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()

# SalePrice ile olan korelasyonu inceleyelim
saleprice_corr = correlation_matrix['SalePrice'].sort_values(ascending=False)
print("SalePrice ile olan korelasyonlar:\n", saleprice_corr)
