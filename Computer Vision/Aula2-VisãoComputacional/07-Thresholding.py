import cv2
import matplotlib.pyplot as plt

img = cv2.imread('Aula2-VisãoComputacional/img/garrafas.jpg', 0)
plt.imshow(img,cmap='gray')

img.max()
img.shape

ret,thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# Qualquer valor abaixo de 127 = 0; acima = 255
plt.imshow(thresh1,cmap='gray')
plt.show()
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# Qualquer valor abaixo de 127 = 0; acima = 255
plt.imshow(thresh1, cmap='gray')


ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
# Qualquer valor abaixo de 127 = 0; acima = 255
plt.imshow(thresh1, cmap='gray')

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# Qualquer valor abaixo de 127 = 0; acima = 255
plt.imshow(thresh1, cmap='gray')

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
# Qualquer valor abaixo de 127 = 0; acima = 255
plt.imshow(thresh1, cmap='gray')
plt.show()
