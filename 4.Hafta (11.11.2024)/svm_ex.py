import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data  
y = iris.target  # etiketlerimiz

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear')  


svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Model doğruluğu: {accuracy * 100:.2f}%")

# ilk iki özelliği almısız burda yani irisin sepal uzunluğu ve genişliği
X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

svm_model.fit(X_train_2d, y_train)

# Karar sınırlarını çizmek için bir ızgara oluşturuyoruz
h = .02  # ızgara adımı
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Tahmin yapalım ve karar sınırlarını çizelim
Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Karar sınırlarını çizecek grafik
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o', s=100)
plt.title("SVM Karar Sınırları")
plt.show()

'''
simdi en önemlisi bence grafiği okumaya çalışalım. Öncelikle sınırlarmız doğrusal şekilde çünkü kerneli linear yaptık
karar sınırlarımıız svmnin ana mantığı olan veriler arasındaki en geniş marjini oluşturacak şekilde yerleştirilir
her nokta bir gözlem sepal uzunluğu falan
renklerse ait olduğu sınıfı gösteriyor
destek vektörerlimiz karar sınırlarının etrafındaki noktalarla çakışır
'''