import cv2
import matplotlib.pyplot as plt

img2 = cv2.imread('Aula2-VisãoComputacional/img/secret.png')
img1 = cv2.imread('Aula2-VisãoComputacional/img/mining.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()

img1.shape
img2.shape

img1 = cv2.resize(img1,(800,800))
img2 = cv2.resize(img2,(800,800))

blended = cv2.addWeighted(src1=img1, alpha=0.8, src2=img2, beta=0.3, gamma=0)

plt.imshow(blended)
plt.show()
