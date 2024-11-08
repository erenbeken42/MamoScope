
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#kredi veri seti oluşturuyoruz
data = {
    'Gelir': [50000, 60000, 35000, 45000, 85000, 90000, 40000, 70000, 55000, 62000],
    'Yaş': [25, 45, 35, 50, 23, 52, 40, 30, 28, 36],
    'Kredi Geçmişi': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1],  # 1: İyi kredi geçmişi, 0: Kötü kredi geçmişi
    'Onay': [1, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # 1: Onaylandı, 0: Reddedildi
}
df = pd.DataFrame(data)

#X in neden büyük harf olduğnuu sordum genelde bağımsız değilkenler olduğu için ve genelde çok boyutlu bir veri yapısını temsil ettiği içinmis
#aynı mantıkla y de  tek sonuç vektörünü temsil ettiği için 
X = df[['Gelir', 'Yaş', 'Kredi Geçmişi']] #iris örneğinde burası tam net değildi simdi cok iyi oturdu
y = df['Onay'] #bağımlı değişken yaani

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#simdi trainler eğitmek için onlar okey 

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# 1 tane test setini gönderip prediciton yapıos
y_pred = dt.predict(X_test)

# sonra ana veriden çektiğmiiz test verisiiyle bizim predict ettiğimiz veriyi karşılaştırıp doğrulunu kontrol ediyoruz oldu tamam
accuracy = accuracy_score(y_test, y_pred)
print("Kredi veri setinde Karar Ağacı ile doğruluk:", accuracy)
