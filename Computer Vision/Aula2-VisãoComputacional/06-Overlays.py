import cv2
import matplotlib.pyplot as plt

img2 = cv2.imread('Aula2-VisãoComputacional/img/secret.png')
img1 = cv2.imread('Aula2-VisãoComputacional/img/mining.jpg')

img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

img_menor = cv2.resize(img2,(1900,1600))

plt.imshow(img_menor)
plt.show()

img_maior = img1

x_offset = 500
y_offset = 500

x_end = x_offset + img_menor.shape[1]
y_end = y_offset + img_menor.shape[0]

# img_menor.shape
# Y , X , COR

img_maior[y_offset:y_end,x_offset:x_end] = img_menor
plt.imshow(img_maior)
plt.show()
