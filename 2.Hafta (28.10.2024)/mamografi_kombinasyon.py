import cv2
import matplotlib.pyplot as plt

image = cv2.imread("mamo3.jpg", cv2.IMREAD_GRAYSCALE)

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

median_blur = cv2.medianBlur(image, 5)

canny_gaussian = cv2.Canny(gaussian_blur, threshold1=100, threshold2=200) # gaussian + canny verisonu

canny_median = cv2.Canny(median_blur, threshold1=100, threshold2=200) #canny + median versiyonu

sobelx_gaussian = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=5)
sobely_gaussian = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=5)
sobel_gaussian = cv2.magnitude(sobelx_gaussian, sobely_gaussian)

sobelx_median = cv2.Sobel(median_blur, cv2.CV_64F, 1, 0, ksize=5)
sobely_median = cv2.Sobel(median_blur, cv2.CV_64F, 0, 1, ksize=5)
sobel_median = cv2.magnitude(sobelx_median, sobely_median)

plt.figure(figsize=(15, 10))

plt.subplot(3, 3, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.title("Gaussian Blur")
plt.imshow(gaussian_blur, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.title("Median Blur")
plt.imshow(median_blur, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.title("Gaussian + Canny Kenar Algılama")
plt.imshow(canny_gaussian, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.title("Median + Canny Kenar Algılama")
plt.imshow(canny_median, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.title("Gaussian + Sobel Kenar Algılama")
plt.imshow(sobel_gaussian, cmap="gray")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.title("Median + Sobel Kenar Algılama")
plt.imshow(sobel_median, cmap="gray")
plt.axis("off")

plt.show()
