import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_img():
    img = cv2.imread('Aula2-VisãoComputacional-20190807T162425Z-001/Aula2-VisãoComputacional/img/parede.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

#load_img()

def display_img(img):
    fig = plt.figure(figsize=(10,8)) # dimensao em polegadas
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.show()

i = load_img()
display_img(i)

gamma = 1/2 # CORREÇÃO GAMMA AJUSTA O BRILHO DA IMAGEM

i
#ATENÇÂO!! INTEIROS!

img_gamma = np.power(i/255,gamma)
display_img(res)

img_gamma

gamma = 8 # QTO MAIOR O GAMMA, MAIS ESCURA FICA A IMAGEM - APROXIMO O PIXEL DO VALOR 255

img_gamma2 = np.power(img_gamma,gamma)
display_img(img_gamma2)

# -----------
# Função Blur
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='NCE',org=(100,600),fontFace=font,fontScale=13,color=(255,0,0),thickness=4)
display_img(img)

kernel = np.ones(shape=(5,5),dtype=np.float32)/25 # argument Shape: dimensao da matriz # quando aumento o tamanho da matriz, ele clareia a imagem

kernel

dst = cv2.filter2D(img,-1,kernel)
display_img(dst)
#-1 = Input Depth

# -----
# Blur do OpenCV
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='NCE',org=(100,600),fontFace=font,fontScale=13,color=(255,0,0),thickness=4)
print('ni')

blurred_cv = cv2.blur(img,ksize=(10,10))
display_img(blurred_cv)

# -----
# GaussianBlur do OpenCV
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='CV',org=(100,600),fontFace=font,fontScale=20,color=(255,0,0),thickness=4)
print('ni')

blurred_gauss = cv2.GaussianBlur(img,(9,9),3)
display_img(blurred_gauss)

# -----
# MedianBlur do OpenCV
img = load_img()
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(img,text='CV',org=(100,600),fontFace=font,fontScale=20,color=(255,0,0),thickness=4)
print('ni')

blur_median = cv2.medianBlur(img,20)
display_img(blur_median)
