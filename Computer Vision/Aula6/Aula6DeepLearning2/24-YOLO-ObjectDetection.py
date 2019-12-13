# # YOLO v3 Object Detection
# 
# FONTE: https://github.com/xiaochus/YOLOv3
# 
# REFERÊNCIA (Trabalho Original - YOLOv3): 

import os
import time
import cv2
import numpy as np
from Aula6.Aula6DeepLearning2.model.yolo_model import YOLO


def process_image(img):
    """Operações de redimensionamento

    # Argumento:
        img: imagem original.

    # Retorna
        image: ndarray(64, 64, 3), imagem processada.
    """
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image


def get_classes(file):
    """Pega o nome das classes.

    # Argumento:
        file: nome das classes.

    # Returns
        class_names: Lista, nome das classes.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Desenha caixas delimitadoras na imagem.

    # Argumento:
        image: imagem original.
        boxes: ndarray, caixas dos objetos.
        classes: ndarray, classes de objetos.
        scores: ndarray, escores de objetos.
        all_classes: todos os nomes das classes.
    """
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()


def detect_image(image, yolo, all_classes):
    """Usa o yolo v3 para detectar imagens.

    # Argumento:
        image: imagem original.
        yolo: YOLO, modelo yolo.
        all_classes: nomes de todas as classes.

    # Retorna:
        image: imagem processada.
    """
    pimage = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        draw(image, boxes, scores, classes, all_classes)

    return image


def detect_video(video, yolo, all_classes):
    """Usa o yolo v3 para detectar video.

    # Argumento:
        video: arquivo de video.
        yolo: YOLO, modelo yolo.
        all_classes: todos os nomes das classes.
    """
    video_path = os.path.join("videos", "test", video)
    camera = cv2.VideoCapture(video_path)
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)

    # Prepare for saving the detected video
    sz = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fourcc = cv2.VideoWriter_fourcc(*'mpeg')

    
    vout = cv2.VideoWriter()
    vout.open(os.path.join("videos", "res", video), fourcc, 20, sz, True)

    while True:
        res, frame = camera.read()

        if not res:
            break

        image = detect_image(frame, yolo, all_classes)
        cv2.imshow("detection", image)

        # Save the video frame by frame
        vout.write(image)

        if cv2.waitKey(110) & 0xff == 27:
                break

    vout.release()
    camera.release()
    


yolo = YOLO(0.6, 0.5)
file = 'Aula6/Aula6DeepLearning2/data/coco_classes.txt'
all_classes = get_classes(file)

# ### Detecting Images

f = 'Aula6/Aula6DeepLearning2/images/test/jingxiang-gao-489454-unsplash.jpg'
path = 'Aula6/Aula6DeepLearning2/images/test'+f
image = cv2.imread(path)
image = detect_image(image, yolo, all_classes)
cv2.imwrite('Aula6/Aula6DeepLearning2/images/res' + f, image)

#%% [markdown]
# # Detecting on Video

#%%
# # detect videos one at a time in videos/test folder    
# video = 'library1.mp4'
# detect_video(video, yolo, all_classes)


