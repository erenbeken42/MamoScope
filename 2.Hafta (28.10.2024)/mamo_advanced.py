import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "mamo3.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=(20, 10))

plt.subplot(3, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Orijinal Görüntü")
plt.axis("off")

hist_eq = cv2.equalizeHist(image) #araştırdığım kadarıyla kontrastı artırıyorki parlaklık daha da yoğunlaşssın
plt.subplot(3, 4, 2)
plt.imshow(hist_eq, cmap="gray")
plt.title("Histogram Eşitleme")
plt.axis("off")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))#gelişmiş bir histogram eşitleme yöntemi.
clahe_img = clahe.apply(image)
plt.subplot(3, 4, 3)
plt.imshow(clahe_img, cmap="gray")
plt.title("CLAHE")
plt.axis("off")

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)
plt.subplot(3, 4, 4)
plt.imshow(gaussian_blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

median_blur = cv2.medianBlur(image, 5)
plt.subplot(3, 4, 5)
plt.imshow(median_blur, cmap="gray")
plt.title("Median Blur")
plt.axis("off")

canny_edges = cv2.Canny(image, 100, 200)
plt.subplot(3, 4, 6)
plt.imshow(canny_edges, cmap="gray")
plt.title("Canny Kenar Algılama (Orijinal)")
plt.axis("off")

canny_clahe = cv2.Canny(clahe_img, 100, 200)
plt.subplot(3, 4, 7)
plt.imshow(canny_clahe, cmap="gray")
plt.title("Canny Kenar Algılama (CLAHE)")
plt.axis("off")

sobelx = cv2.Sobel(clahe_img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(clahe_img, cv2.CV_64F, 0, 1, ksize=5)
sobel_edges = cv2.magnitude(sobelx, sobely)
plt.subplot(3, 4, 8)
plt.imshow(sobel_edges, cmap="gray")
plt.title("Sobel Kenar Algılama (CLAHE)")
plt.axis("off")

_, thresh_simple = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)#araştırıken buldum denemelik 
plt.subplot(3, 4, 9)
plt.imshow(thresh_simple, cmap="gray")
plt.title("Basit Eşikleme Segmentasyonu")
plt.axis("off")

_, thresh_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #araştırıken buldum denemelik 
plt.subplot(3, 4, 10)
plt.imshow(thresh_otsu, cmap="gray")
plt.title("Otsu Eşikleme Segmentasyonu")
plt.axis("off")

image_flat = image.reshape((-1, 1))
image_flat = np.float32(image_flat)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv2.kmeans(image_flat, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
segmented_image = labels.reshape(image.shape) #araştırıken buldum denemelik 
plt.subplot(3, 4, 11)
plt.imshow(segmented_image, cmap="gray")
plt.title("K-Means Segmentasyonu")
plt.axis("off")

contours, _ = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#araştırıken buldum denemelik 
contour_img = np.zeros_like(image)
cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 1)
plt.subplot(3, 4, 12)
plt.imshow(contour_img, cmap="gray")
plt.title("Kontur Algılama (Otsu)")
plt.axis("off")

plt.tight_layout()
plt.show()
