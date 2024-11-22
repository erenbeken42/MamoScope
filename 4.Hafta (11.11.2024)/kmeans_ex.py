import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# random bir veri seti çektik sample-center zaten belline olduğu random zaten rastgelelilk, std= standartsapma küçülmesi olmaı kümeleri sıkılaştırı
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=42)

# bu kısım gösterim kısmı s ile vveri noktalarının büyüklüğü ayarlanır 
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Veri Seti")
plt.show()

# cluster sayısı yani veriyi 4 kümeye ayıracağız mesela bu sayıyı 5 yapalım 1 kümede 2 veri merkezi oldu bu istemediğimiz bir durum 
# olabilir o yüzden bu ayarlamaları tutarlı yapmamız lazım rastgellilik olayının sabit olmasını zaten önceki kodlarımda mantığını anladım
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X) #x üzerijnden eğiteceğiz modelimizi

# Küme merkezlerini centroidse kayıt ettik ki gösterim kısmında işaretleyebileim
centroids = kmeans.cluster_centers_

# her veri noktalarının hangi kümeye atandığını belirten listeyi döndürüyormuş
labels = kmeans.labels_

# görsellişttirme 
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Küme Merkezleri')
plt.title("K-Means Kümeleme Sonuçları")
plt.legend()
plt.show()
