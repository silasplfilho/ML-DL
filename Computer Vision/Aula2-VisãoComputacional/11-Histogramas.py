import cv2
import matplotlib.pyplot as plt
# ----
nat = cv2.imread('Aula2-VisãoComputacional/img/nature.jpg')
show_nat = cv2.cvtColor(nat, cv2.COLOR_BGR2RGB)

parede = cv2.imread('Aula2-VisãoComputacional/img/parede.jpg')
show_parede = cv2.cvtColor(parede, cv2.COLOR_BGR2RGB)

rainbow = cv2.imread('Aula2-VisãoComputacional/img/rainbow.jpg')
show_rainbow = cv2.cvtColor(rainbow, cv2.COLOR_BGR2RGB)
# ----


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


# OPENCV = BGR
hist = cv2.calcHist([parede], channels=[2], mask=None, histSize=[256], ranges=[0, 256])
hist.shape
plt.plot(hist)
plt.show()

img = parede

cores = ('b', 'g', 'r')

for i, col in enumerate(cores):
    histr = cv2.calcHist([img], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
    plt.ylim([0, 15000])

plt.title('Histograma')
plt.show()
