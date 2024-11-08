from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# maakine öğrenmesinin en klasik örneği iris çiçeği;
iris = load_iris()
X = iris.data  # tahmin yaparken kullanacağı özellikler
y = iris.target  # tahmin etmeye çalıştığı şey

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)#yüzde 30nu test alacak 
#yüzde 70niyle algoritma besleyecek random_state=herhangi bir sayı olabilr aynı değer kullanıldığında aynı değer elde edilebilir
#farklı değer kullanıldığında bölğnme şekli değişiyor ya işte minecrafttaki seed mantığının aynısı randomluk katıyor oke

#kcük ve basit düzeydeki veri setleriyle çalışırken  KNN uygun hem öğrenmesi hem pratiği kolaymıs ondan sectik
knn = KNeighborsClassifier(n_neighbors=3) # 3 burda en yakın 3 komsuya göre sınıflandırıcı olustruyor. 
knn.fit(X_train, y_train) #fit ile modeli eğitim verisiyle eğitir (yani veri noktalarını ve sınıflarını kaydeder)

# Test seti ile tahmin yapalım
y_pred = knn.predict(X_test)#simdi sistem su eğitim verisi algoritma test verisi suanda test verisini algoya soktuk ve tahmin çıkarttık

#accuracy_score, modelin tahmin ettiği etiketlerin (y_pred) gerçek etiketlerle (y_test) ne kadar uyumlu olduğunu gösteri
accuracy = accuracy_score(y_test, y_pred)
print("İris veri setinde KNN ile doğruluk:", accuracy) #Doğruluk doğru tahmin edilen sınıfların sayısının tüm tahmin edilen sınıflara oranıdır.
