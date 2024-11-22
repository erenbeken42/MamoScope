from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = load_iris()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#estimator karar ağaçlarını sayısını belirliyoz
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Dogruluk: {accuracy:.2f}")#doğru tahminlerin tüm tahminlere oranı 
print(f"Kesinlik: {precision:.2f}")#tru pozitifin tüm pozitiflere oranı (olasılık dersindeki olay)
print(f"Duyarlılık: {recall:.2f}")#TP/TP+FN
print(f"F1-Skoru: {f1:.2f}")#precision ve recall'un harmonik ortalamasıdır.

print("\n Ayrıntılı  Rapor:")
print(classification_report(y_test, y_pred, target_names=data.target_names))