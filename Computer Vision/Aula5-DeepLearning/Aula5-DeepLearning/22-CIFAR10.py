# # CIFAR-10
# O dataset CIFAR-10 consiste em 10 tipos de imagens diferentes, ou classes
# -----
# # Dados
# O CIFAR-10 é um dataset composto por 50K imagens de 32x32, coloridas, rotuladas
# em 10 categorias e 10K imagens teste.

# Suprimir Warnings
import warnings
warnings.filterwarnings('ignore')
from keras.datasets import cifar10

(x_treina, y_treina), (x_teste, y_teste) = cifar10.load_data()
x_treina.shape
x_teste.shape
x_treina[0].shape

import matplotlib.pyplot as plt

# SAPO - FROG
plt.imshow(x_treina[0])
plt.show()

# BARCO - SHIP
plt.imshow(x_treina[100])
plt.show()

# # Pré-Processamento
x_treina[0]
x_treina[0].shape

x_treina.max()

x_treina = x_treina/255
x_teste = x_teste/255
x_treina.shape

x_teste.shape

# ## Labels
from keras.utils import to_categorical
y_treina.shape
y_treina[0]

y_cat_treina = to_categorical(y_treina, 10)
y_cat_treina.shape
y_cat_treina[0]

y_cat_teste = to_categorical(y_teste, 10)

# ----------
# # Criando o Modelo
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

modelo = Sequential()

## Primeiro conjunto de camadas
# CAMADA CONVOLUCIONAL
modelo.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu',))
# CAMADA DE POOLING
modelo.add(MaxPool2D(pool_size=(2, 2)))

## Segundo conjunto de camadas
# CAMADA CONVOLUCIONAL
modelo.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu',))
# CAMADA DE POOLING
modelo.add(MaxPool2D(pool_size=(2, 2)))

# FLATTENING - 28x28 para 764 ANTES DA ÚLTIMA CAMADA
modelo.add(Flatten())

# 256 NEURONS NA CAMADA OCULTA
modelo.add(Dense(256, activation='relu'))

# ÚLTIMA CAMADA É O CLASSIFICADOR, 10 CLASSES
modelo.add(Dense(10, activation='softmax'))

modelo.compile(loss='categorical_crossentropy',
               optimizer='rmsprop',
               metrics=['accuracy'])

modelo.summary()

#HORA DO CAFÉ!!
modelo.fit(x_treina, y_cat_treina, verbose=1, epochs=10)

# Descomente e rode para salvar o modelo treinado.
# model.save('cifar_10epochs.h5')

model.metrics_names

model.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_test)

print(classification_report(y_test,predictions))

# ## Para Casa: Modelo Grande

modelofull = Sequential()

## Primeiro conjunto de camadas

# CAMADA CONVOLUCIONAL
modelofull.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# CAMADA CONVOLUCIONAL
modelofull.add(Conv2D(filters=32, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))

# CAMADA DE POOLING
modelofull.add(MaxPool2D(pool_size=(2, 2)))

## Segundo conjunto de camadas

# CAMADA CONVOLUCIONAL
modelofull.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))
# CAMADA CONVOLUCIONAL
modelofull.add(Conv2D(filters=64, kernel_size=(4,4),input_shape=(32, 32, 3), activation='relu',))

# CAMADA DE POOLING
modelofull.add(MaxPool2D(pool_size=(2, 2)))

# FLATTENNING 28x28 para 764
modelofull.add(Flatten())

# 512 NEURONS NA CAMADA OCULTA (Experimente alterar)
modelofull.add(Dense(512, activation='relu'))

# ÚLTIMA CAMADA - CLASSIFICADOR - 10 CLASSES
modelofull.add(Dense(10, activation='softmax'))


modelofull.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


modelofull.fit(x_treina,y_cat_treina,verbose=1,epochs=20)

modelofull.evaluate(x_test,y_cat_test)

from sklearn.metrics import classification_report

predictions = model.predict_classes(x_teste)

print(classification_report(y_teste,predictions))

modelofull.save('CIFAR10_FULL_model.h5')
