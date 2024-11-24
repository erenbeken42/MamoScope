import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('bcw.csv')

print(df.isnull().sum()) 

df = df.drop(['id', 'Unnamed: 32'], axis=1)
#isimize yaramayanları çıkarıyorum

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
#burası kritik belirdeliğimiz özelliği numerikleştiriyorzuki sınıflandırmamız daha kolay olsun

#hedef özelliğimiz teşhiş(diagnosis)
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# yüzde 80 eğitim 20 test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# standardize etme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression modeli
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Performans metriklerini hesaplama
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
