from PIL import Image
from PIL import ImageFilter

img = Image.open("mamo3.jpg")

img.save("yeni_goruntu.png", "PNG")

img_resized = img.resize((200, 200))

img_resized.save("kucultulmus_goruntu.jpg")

img_cropped = img.crop((50, 50, 200, 200))

img_cropped.save("kirpilmis_goruntu.jpg")

img_rotated = img.rotate(90)

img_rotated.save("dondurulmus_goruntu.jpg")

img_gray = img.convert("L")
img_gray.save("gri_tonlama.jpg")

img_bw = img.convert("1")
img_bw.save("siyah_beyaz.jpg")

img_blur = img.filter(ImageFilter.BLUR)
img_blur.save("bulanık_goruntu.jpg")

img_edge = img.filter(ImageFilter.FIND_EDGES) #kenar tesptipi mamografi teşhislerinde kenarları olan yoğun beyazlıkların
# tümör riski yüksekikmil bu fonksiyonu unutma!
img_edge.save("kenar_goruntu.jpg")

width, height = img.size #boyut alma mantığı yine cv2 deki gibi

for x in range(width):
    for y in range(height):
        r, g, b = img.getpixel((x, y))
        img.putpixel((x, y), (r, 0, 0)) # tüm pikselleri kapsadı yukarda burda da her pikselin yeşil ve mavi verisini sıfırladı damnn!!

img.save("kirmizi_tonlar.jpg")

