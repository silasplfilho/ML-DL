# # CNNs para Classificação de Imagens - MNIST
# Suprimir Warnings
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.datasets import mnist

(x_treina, y_treina), (x_teste, y_teste) = mnist.load_data()

# ##  Visualizando os Dados
x_treina.shape

imagem = x_treina[0]
imagem
imagem.shape

plt.imshow(imagem)
plt.show()

# # Pré-Processamento
# Precisamos fazer com que nossas labels sejam compreendidas por nossa CNN.
# ## Labels
y_treina
y_teste

# Aparentemente, as labels correspondem aos números. Precisamos traduzir isto para
# "one hot encoding", de forma que nossa CNN seja capa de compreender.
# Senão, ela entenderá como um problema de regressão. O Keras tem uma função fácil de usar
# para "one hot encoding":

from keras.utils.np_utils import to_categorical
y_treina.shape
y_exemplo = to_categorical(y_treina)
y_exemplo

y_exemplo.shape
y_exemplo[0]

y_cat_teste = to_categorical(y_teste, 10)
y_cat_treina = to_categorical(y_treina, 10)

# ### Processando dados X
# Devemos normalizar os dados em X.
imagem.max()
imagem.min()

x_treina = x_treina/255
x_teste = x_teste/255

scaled_imagem = x_treina[0]
scaled_imagem.max()

plt.imshow(scaled_imagem)
plt.show()

# ## Remodelando os Dados
# Nossos dados compreendem 60.000 imagens armazenadas em um array 28 por 28.
# Está correto para uma CNN, mas precisamos adicionar uma dimensão a mais para mostrar que
# estamos lidando com 1 canal RGB (uma vez que tecnicamente, as imagens estão em preto e branco,
# mostrando valores de 0-255 em um canal), uma imagem colorida teria 3 dimensões.

x_treina.shape
x_teste.shape

# Reshape para incluir a dimensão de canal (1 canal, no caso)
x_treina = x_treina.reshape(60000, 28, 28, 1)
x_treina.shape
x_teste = x_teste.reshape(10000, 28, 28, 1)
x_teste.shape

# # Treinando o Modelo
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

modelo = Sequential()

# CAMADA CONVOLUCIONAL
modelo.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu',))
# CAMADA DE POOLING
modelo.add(MaxPool2D(pool_size=(2, 2)))

# EXECUTA FLATTEN DAS IMAGENS, de 28x28 para 764 ANTES DA ÚLTIMA CAMADA
modelo.add(Flatten())

# 128 NEURONS NA CAMADA OCULTA
modelo.add(Dense(128, activation='relu'))

# ÚLTIMA CAMADA É O CLASSIFICADOR EM SI, 10 POSSÍVEIS CLASSES.
modelo.add(Dense(10, activation='softmax'))


modelo.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


modelo.summary()

# ## Treinando o Modelo

# HORA DO CAFẼ!!! ISTO VAI DEMORAR.
# ALTERE AS EPOCHS SE NECESSÁRIO
# SUA ACURÁCIA PODE SER MENOR EM UMA CPU EM RELAÇÃO AO TREINAMENTO EM GPU
modelo.fit(x_treina, y_cat_treina, epochs=2)

# ## Testando o Modelo
modelo.metrics_names
modelo.evaluate(x_teste, y_cat_teste)

from sklearn.metrics import classification_report

predictions = modelo.predict_classes(x_teste)
y_cat_teste.shape
y_cat_teste[0]
predictions[0]

y_teste

print(classification_report(y_teste, predictions))
