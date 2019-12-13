import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_img():
    img = np.zeros((600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text='HELLO', org=(50, 300),
                fontFace=font, fontScale=5,
                color=(255, 255, 255), thickness=26)
    return img


def display_img(img):  # acrescentar na funcao **args - para q assim eu tenha plots de mais de uma imagem
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()

img = load_img()
display_img(img)

kernel = np.ones((5, 5), dtype=np.uint8)
kernel
# -----
# Erosao
res = cv2.erode(img, kernel, iterations=6)
display_img(res)

img = load_img()
# criando ruido branco
w_noise = np.random.randint(low=0, high=2, size=(600, 600))
w_noise
display_img(w_noise)

img.max()

w_noise = w_noise * 255
w_noise
display_img(w_noise)

img_noise = w_noise + img
display_img(img_noise)

# ------
# aplicando morfologia opening (erosion + dilation)
opening = cv2.morphologyEx(img_noise, cv2.MORPH_OPEN, kernel)
display_img(opening)

img = load_img()

# usando opening numa imagem c/ ruido preto
## preparando o ruido preto (white noise dentro da letra)
b_noise = np.random.randint(low=0, high=2, size=(600, 600))
b_noise = b_noise * -255

b_noise_img = img + b_noise
b_noise_img[b_noise_img == -255] = 0
display_img(b_noise_img)
b_noise_img.max()

## aplicando morfologia closing
closing = cv2.morphologyEx(b_noise_img, cv2.MORPH_CLOSE, kernel)
display_img(closing)
display_img(img)

img = load_img()

## aplicando morfologia gradient (Erosion - Dilation)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

display_img(gradient)