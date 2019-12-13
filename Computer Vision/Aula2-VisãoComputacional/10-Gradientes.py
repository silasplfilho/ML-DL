import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

img = cv2.imread('Aula2-VisãoComputacional/img/sudoku.jpg', 0)

display_img(img)
# ----
sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=5)

display_img(sobelx)
# ----
sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=5)

display_img(sobely)
# ----
laplace = cv2.Laplacian(img, cv2.CV_32F)

display_img(laplace)
# ----
blend = cv2.addWeighted(src1=sobelx, alpha=0.5, src2=sobely, beta=0.5, gamma=0)

display_img(blend)
# ----
ret,th1 = cv2.threshold(blend,100,255,cv2.THRESH_BINARY)
display_img(th1)
# ----
kernel = np.ones((4,4),np.uint8)

gradient = cv2.morphologyEx(blend,cv2.MORPH_GRADIENT, kernel)

display_img(gradient)
