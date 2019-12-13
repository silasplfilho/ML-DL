import cv2
import numpy as np
import matplotlib.pyplot as plt
# ----
xadrez = cv2.imread('Aula3-MachineLearning/xadrez.png')
xadrez = cv2.cvtColor(xadrez, cv2.COLOR_BGR2RGB)
# ----
gray_xadrez = cv2.cvtColor(xadrez, cv2.COLOR_RGB2GRAY)

tabuleiro = cv2.imread('Aula3-MachineLearning/tabuleiro.jpg')
tabuleiro = cv2.cvtColor(tabuleiro, cv2.COLOR_BGR2RGB)
# ----
gray_tabuleiro = cv2.cvtColor(tabuleiro, cv2.COLOR_RGB2GRAY)

plt.imshow(gray_tabuleiro)
plt.show()

plt.imshow(gray_xadrez)
plt.show()
# ----
# Harris-Corner Detection

gray = np.float32(gray_xadrez)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

xadrez[dst > 0.01*dst.max()] = [255, 0, 0]

plt.imshow(xadrez)
plt.show()
# ----
# Harris-Corner Detection

gray = np.float32(gray_tabuleiro)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(dst, None)

tabuleiro[dst > 0.01*dst.max()] = [255, 0, 0]

plt.imshow(tabuleiro)
plt.show()
# ----
# Shi-Tomasi

corners = cv2.goodFeaturesToTrack(gray_tabuleiro, 0, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(tabuleiro, (x, y), 3, (255, 0, 0), -1)

plt.imshow(tabuleiro)
plt.show()
