import numpy as np
import sklearn
# Suprimir Warnings
import warnings
warnings.filterwarnings('ignore')

# ## Dataset
# Usaremos um dataset de cédulas, os dados são características de imagem derivadas de imagens 
# reais de 400x400 pixels. Notem que o dataset não consiste em imagens reais, outrossim 
# características (features) das imagens.
# _____
# Mais informações:
# 
# https://archive.ics.uci.edu/ml/datasets/banknote+authentication
# ## Carregando o Dataset
# Os dados estão na pasta 'dados', carregue de acordo.

from numpy import genfromtxt
dados = genfromtxt('Aula5-DeepLearning/Aula5-DeepLearning/dados/bank_note_data.txt', delimiter=',')

dados

labels = dados[:,4]

labels

features = dados[:,0:4]

features

X = features
y = labels

# ## Dividindo os Dados entre Treinamento e Teste
# Agora dividiremos os dados entre dois sets - treina e teste. Na vida real, temos um set 
# adicional: validação. Mas iremos simplificar as coisas por enquanto...

from sklearn.model_selection import train_test_split

X_treina, X_teste, y_treina, y_teste = train_test_split(X, y, test_size=0.33, random_state=42)

X_treina
X_teste
y_treina
y_teste

# ## Padronizando os Dados
# Normalmente, você garante melhor performance quando padronia os dados. Padronizar normalmente
# consiste em normalizar os valores de modo a caber em um certo range, como 0 a 1 ou -1 a 1.
# A bibilioteca Scikit Learn tem uma função para isso:
# 
# http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html

from sklearn.preprocessing import MinMaxScaler

scaler_object = MinMaxScaler()
scaler_object.fit(X_treina)

scaled_X_treina = scaler_object.transform(X_treina)
scaled_X_teste = scaler_object.transform(X_teste)

# Agora temos os dados normalizados!
X_treina.max()

scaled_X_treina.max()

X_treina
scaled_X_treina

# ## Criando a Rede Neural com o Keras
from keras.models import Sequential
from keras.layers import Dense

# Cria o modelo
modelo = Sequential()
# 8 Neurons, com 4 características de entrada.
modelo.add(Dense(4, input_dim=4, activation='relu'))
# Adiciona outra camada densamente conectada
modelo.add(Dense(8, activation='relu'))
# Última camada é uma função sigmóide com saída 0 ou 1 (label)
modelo.add(Dense(1, activation='sigmoid'))

# ### Compilar o Modelo
modelo.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# ## Treina o Modelo
# Teste outros valores de iteração - epochs
modelo.fit(scaled_X_treina, y_treina, epochs=50, verbose=2)

# ## Prevendo novos dados
# Vamos ver como nosso modelo se comporta prevendo **novos dados**. Lembrando que nosso modelo
# nunca viu os dados testes! Trata-se do mesmo processo com novos dados. Por exemplo, uma nova
# cedula a ser analisada.

scaled_X_teste

# "Cospe" probabilidades por default.
# model.predict(scaled_X_teste)

modelo.predict_classes(scaled_X_teste)

# # Testando a Performance do Modelo
modelo.metrics_names
modelo.evaluate(x=scaled_X_teste, y=y_teste)
# ----------
from sklearn.metrics import confusion_matrix, classification_report

predictions = modelo.predict_classes(scaled_X_teste)

confusion_matrix(y_teste, predictions)

print(classification_report(y_teste, predictions))

# ## Salvando e Carregando Modelos
# 
# Agora que temos um modelo treinado, vamos ver como salvar e carregar.
modelo.save('meumodelo.h5')

from keras.models import load_model

novomodelo = load_model('meumodelo.h5')

novomodelo.predict_classes(X_teste)


