import cv2
image = cv2.imread("mamo3.jpg")

cv2.imshow("çıktı", image)

(y, g) = image.shape[:2] #şimdi burda 2 değer döndürüyor yükseklik ve genişlik
merkez = (g // 2, y // 2)
rotation_matrix = cv2.getRotationMatrix2D(merkez, 45, 1.0) #önce döndürmek istediğimiz fonksiyonu giriyoruz 45 derece, 1 de ölçek
rotated_image = cv2.warpAffine(image, rotation_matrix, (g, y)) #imagei bu fonksiyona göre döndürüyoruz 
cv2.imshow("Döndürülmüş Görüntü", rotated_image)

resized_image = cv2.resize(image, (300, 200))
cv2.imshow("Boyutlandırılmış Görüntü", resized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
