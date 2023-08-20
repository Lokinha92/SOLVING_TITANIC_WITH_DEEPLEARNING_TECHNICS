import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential #Classificador
from keras.layers import Dense, Dropout, PReLU # Neuronio
import matplotlib.pyplot as plt
import seaborn as sns

caminho = "./dados/train.csv"

train = pd.read_csv(caminho)

mediana = 28
train['Age'].fillna(mediana, inplace=True)

moda = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train.Cabin = moda.fit_transform(train[['Cabin']])

train['Embarked'].fillna('S', inplace=True)

train.info()

label = LabelEncoder()

train['Sex'] = label.fit_transform(train['Sex'])

train['Embarked'] = label.fit_transform(train['Embarked'])

atributos = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

classe = train['Survived']

x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, classe, test_size=0.3, random_state=0)

x_treino = tf.convert_to_tensor(x_treino, dtype=tf.float32)
y_treino = tf.convert_to_tensor(y_treino, dtype=tf.float32)
x_teste = tf.convert_to_tensor(x_teste, dtype=tf.float32)
y_teste = tf.convert_to_tensor(y_teste, dtype=tf.float32)

normalizador = StandardScaler()

x_treino = normalizador.fit_transform(x_treino)
x_teste = normalizador.fit_transform(x_teste)

#y_treino = to_categorical(y_treino, 2)
#y_teste = to_categorical(y_teste, 2)

rede = Sequential()

# 6 entradas -> 2 neuronios, 2 neuronios, 2 neuronios, 1 de saída pq é binario

rede.add(Dense(units=2, kernel_initializer='uniform', input_dim=6))
rede.add(PReLU())  # Função de ativação PReLU
rede.add(Dropout(0.2))
rede.add(Dense(units=2, kernel_initializer='uniform'))
rede.add(PReLU())  # Função de ativação PReLU
rede.add(Dropout(0.2))
rede.add(Dense(units=2, kernel_initializer='uniform'))
rede.add(PReLU())  # Função de ativação PReLU
rede.add(Dropout(0.2))

rede.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

rede.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

rede.summary()

treinamento = rede.fit(x_treino, y_treino, batch_size=10 ,epochs=500, validation_data= (x_teste, y_teste))

# previsoes

previsoes = rede.predict(x_teste)

previsoes = (previsoes > 0.55)

# performance do modelo no treino

treinamento.history.keys()
plt.plot(treinamento.history['val_loss']) # evolução do erro

plt.plot(treinamento.history['val_accuracy'])

# matriz de confusao

confusao = confusion_matrix(y_teste, previsoes)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão')
plt.show()

# submetendo aos dados de teste:

caminhoteste = "./dados/test.csv"

test = pd.read_csv(caminhoteste)

test['Age'].fillna(mediana, inplace=True)
test[['Cabin']] = moda.transform(test[['Cabin']])
test['Embarked'].fillna('S', inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

test.info()

labeltest = LabelEncoder()

test['Sex'] = labeltest.fit_transform(test['Sex'])
test['Embarked'] = labeltest.fit_transform(test['Embarked'])

atributosteste = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

atributosteste = atributosteste.values

atributosteste = tf.convert_to_tensor(atributosteste, dtype=tf.float32)

atributosteste = normalizador.fit_transform(atributosteste)


previsaoTeste = rede.predict(atributosteste)

previsaoTeste = (previsaoTeste > 0.55)


previsaoTeste = previsaoTeste.flatten()
salvar = pd.DataFrame({'PassengerID': test['PassengerId'], 'Name': test['Name'], 'Survived': previsaoTeste})
salvar.to_csv('Resultado.csv', index=False)