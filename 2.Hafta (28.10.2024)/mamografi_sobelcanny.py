import cv2
import matplotlib.pyplot as plt

image = cv2.imread("mamo3.jpg", cv2.IMREAD_GRAYSCALE)

# Cannyli olan
canny_edges = cv2.Canny(image, threshold1=100, threshold2=200)

# Sobeli olan
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # X yönünde Sobel
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Y yönünde Sobel
sobel_edges = cv2.magnitude(sobelx, sobely)

plt.figure(figsize=(10, 5)) #görselleştirmek için önce bir tabanı yani figüre yaratıyoruz 

plt.subplot(1, 3, 1) #1 satır 3 sütuna böldük ve 1.sütunu seçtik
plt.title("Orijinal Görüntü") #og yi seçtik
plt.imshow(image,cmap="gray")# gri tonlamalı seçmezsek renkli çıktı veriyor tam vurgulayamıyoruz siyah beyaza göre 
plt.axis("off") #koordinat sistemini kapattım çünkü sayıların gözükmesine gerek yok yanlarda

plt.subplot(1, 3, 2)# yine aynı sekilde böldük 2.sütunu seçtim
plt.title("Canny Kenar Algılama")
plt.imshow(canny_edges, cmap="gray")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Sobel Kenar Algılama")
plt.imshow(sobel_edges, cmap="gray")
plt.axis("off")

plt.show()
