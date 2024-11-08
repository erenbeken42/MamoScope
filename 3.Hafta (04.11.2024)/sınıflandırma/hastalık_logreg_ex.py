import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = {
    'Yaş': [25, 45, 35, 50, 23, 52, 40, 30, 28, 36],
    'BMI': [22.4, 27.5, 30.1, 26.7, 24.3, 32.1, 28.4, 23.6, 25.9, 29.2],
    'Hastalık': [0, 1, 1, 1, 0, 1, 1, 0, 0, 1]  # 1: Hasta, 0: Sağlıklı
}
df = pd.DataFrame(data)

X = df[['Yaş', 'BMI']]
y = df['Hastalık']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Bu yöntem, genellikle sürekli değerleri tahmin etmek için kullanılır. 
#Örneğin, ev fiyatlarını tahmin etmek, bir kişinin gelirini tahmin etmek gibi durumlarda regresyon kullanılır.hmm cok net değil bakalm
#Lojistik regresyon genellikle ikili sınıflandırma (binary classification) problemleri için kullanılır.
#Örneğin, bir kişinin hasta olup olmadığını tahmin etmek, bir e-posta mesajının spam olup olmadığını belirlemek gibi.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)


y_pred = log_reg.predict(X_test)

#Lojistik regresyon, ismiyle regresyoni çağrıştırsa da aslında bir sınıflandırma yöntemidir. 
#Bu teknik, bağımlı değişkenin sürekli değil, kategorik olduğunu ve iki ya da daha fazla sınıfa ait olabileceğini göz önünde bulundurur. 
#Bu nedenle lojistik regresyon, regresyonun sınıflandırma problemine uyarlanmış bir versiyonudur.
accuracy = accuracy_score(y_test, y_pred)
print("Hastalık veri setinde Lojistik Regresyon ile doğruluk:", accuracy)

#bu olasılık dersindeki true postive false postivie tru negatvie false negative olayı 
cm = confusion_matrix(y_test, y_pred)
print("Karmaşıklık Matrisi:\n", cm)