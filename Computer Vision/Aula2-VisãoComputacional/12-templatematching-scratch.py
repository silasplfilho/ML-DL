import cv2
import matplotlib.pyplot as plt
# ----
full = cv2.imread('Aula2-VisãoComputacional/img/lula.jpg')
full = cv2.cvtColor(full, cv2.COLOR_BGR2RGB)
# ----
# def display_img(img):
#     fig = plt.figure(figsize=(12, 10))
#     ax = fig.add_subplot(111)
#     ax.imshow(img, cmap='gray')
#     plt.show()


# OPENCV = BGR
face = full[120:245, 300:400]

methods = ["cv2.TM_CCOEFF", "cv2.TM_CCOEFF_NORMED", "cv2.TM_CCOR", "cv2.TM_SQDIFF", 
           "cv2.TM_SQDIFF_NORMED"]
my_method = eval("cv2.TM_CCOEFF")

res = cv2.matchTemplate(full, face, my_method)
plt.imshow(res)
plt.imshow(full)
plt.show()

# for i in methods:
#     full_copy = full.copy()

#     method = i
