<h1 align='center'> SOLVING TITANIC WITH DEEP LEARNING TECHNICS </h1>

<strong><p align = center> GUSTAVO HENRIQUE D'ANUNCIAÇÃO FERREIRA</p></strong>

<h2 align = 'center'> ❓ Resumo </h2>

<b>Esta implementação propõe uma solução para o clássico problema de classificação do conjunto de dados do Titanic utilizando um modelo de Rede Neural Artificial.</b>

<h2 align = 'center'> 🛥️ O CONJUNTO DE DADOS DO TITANIC </h2>

O conjunto de dados do Titanic é um conjunto de dados clássico amplamente utilizado em treinamento e aprendizado de máquina. Ele contém informações sobre os passageiros do famoso navio RMS Titanic, que naufragou em sua viagem inaugural em 15 de abril de 1912, após colidir com um iceberg. O conjunto de dados é frequentemente usado para tarefas de classificação e análise preditiva, especialmente para prever se um passageiro sobreviveu ou não ao desastre.

O conjunto contém informações sobre 891 passageiros do Titanic. As informações são dividadas em 11 atributos diferentes e uma classe (Survived):

- PassengerId: Um identificador único para cada passageiro.
- Pclass (Classe): A classe do bilhete do passageiro (1ª, 2ª ou 3ª classe).
- Name: Nome do passageiro.
- Sex: Gênero do passageiro (masculino ou feminino).
- Age: Idade do passageiro.
- SibSp: (Número de Irmãos/Cônjuges a Bordo): Indica quantos irmãos/cônjuges o passageiro tinha a bordo.
- Parch: (Número de Pais/Filhos a Bordo): Indica quantos pais/filhos o passageiro tinha a bordo.
- Ticket: Número do bilhete.
- Fare: (Tarifa): O valor pago pelo passageiro pela passagem.
- Cabin: (Cabine): O número da cabine do passageiro.
- Embarked: (Local de Embarque): O local onde o passageiro embarcou (C = Cherbourg, Q = Queenstown, S = Southampton).
- Survived: Indica se o passageiro sobreviveu (1) ou não (0) ao desastre.

Este é o cabeçalho do conjunto de dados de treino:

<div align = center> <img align src = /img/trainhead.png> </div>

<h2 align = 'center'>🧩IMPLEMENTAÇÃO:</h2>

- As seguintes bibliotecas foram usadas durante a implementação:

```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, PReLU 
import matplotlib.pyplot as plt
import seaborn as sns
```

- Primeiramente, a leitura dos dados de treino é feita, juntamente com um tratamento dos valores faltantes.

```python
caminho = "./drive/MyDrive/TITANIC/dados/train.csv"

train = pd.read_csv(caminho)

mediana = 28
train['Age'].fillna(mediana, inplace=True)

moda = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
train.Cabin = moda.fit_transform(train[['Cabin']])

train['Embarked'].fillna('S', inplace=True)

train.info()
```

A variável caminho contém o caminho para o arquivo contendo dados de treino, caso eles sejam movidos, o caminho deve ser atualizado.

O tratamento dos dados faltantes ocorre nos atributos 'Age', 'Cabin' e 'Embarked'. Após o tratamento, o método ".info()" é chamado para verificar se o tratamento ocorreu de forma correta.

<div align = center> <img align src = /img/info.png> </div>

O método info() retorna a tabela acima. Note que, não há registros nulos em quaisquer colunas do dataset.

- A seguir, as variáveis categóricas que serão usadas como atributos previsores são tratadas com o LabelEncoder:

```python
label = LabelEncoder()

train['Sex'] = label.fit_transform(train['Sex'])

train['Embarked'] = label.fit_transform(train['Embarked'])
```

O LabelEncoder trata as variáveis categóricas para associar cada registro único a um número inteiro.

- Nessa etapa, os atributos e a classe são selecionados e armazenados nos objetos "atributos" e "classe", e os dados de treino e teste são divididos também.

```python
atributos = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

classe = train['Survived']

x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, classe, test_size=0.3, random_state=0)
```

Neste caso, os dados de treino representam 70% dos dados, os outros 30% ficam para os dados de teste.

- Aqui, a ultima parte de pré-processamento dos dados de treino acontece:

```python
x_treino = tf.convert_to_tensor(x_treino, dtype=tf.float32)
y_treino = tf.convert_to_tensor(y_treino, dtype=tf.float32)
x_teste = tf.convert_to_tensor(x_teste, dtype=tf.float32)
y_teste = tf.convert_to_tensor(y_teste, dtype=tf.float32)

normalizador = StandardScaler()

x_treino = normalizador.fit_transform(x_treino)
x_teste = normalizador.fit_transform(x_teste)
```

Os dados de treino e teste são convertidos para tensor_array, e o tipo de dados é convertido para float32 para que a normalização dos atributos de treino e teste com z-score aconteça de forma correta.

- Finalmente, o modelo é criado, configurado, compilado e treinado:

```python
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
```

A estrutura da rede é formada por uma camada de entrada contendo 6 atributos, 3 camadas ocultas com 2 neurônios cada e utilizam a função de ativação PReLU.

Para a camada de saída, temos apenas 1 neurônio, e a função de ativação "sigmoid" é utilizada, já que é um problema de classificação binária.

Na fase de compilação, o optimizador usado foi o 'adam', a função de perda utilizada foi a 'binary_crossentropy' e o 'accuracy' foi usado para calcular as metricas de erro.

Enfim, a rede é treinada através do método .fit passando os dados de treino como os 2 primeiros hiperparâmetros e o número de epochs (vezes que os dados são submetidos à RNA) pode ser alterado ajustando o hiperparâmetro "epochs". Como padrão, o número de epochs foi definido como 500, mas o modelo pode ser treinado com qualquer número de epochs.

Não é necessário criar um bloco de código com o calculo das métricas de erro, já que o modelo de RNA já faz esse calculo durante o treinamento, basta passar uma tupla contendo os dados de teste (respectivamente) para o hiperparâmetro "validation_data"

- As previsoes são feitas. Depois, uma visualização gráfica da performance do modelo no treinamento e a matriz de confusão com a performance das previsões são mostradas.

```python
# previsoes

previsoes = rede.predict(x_teste)

previsoes = (previsoes > 0.55)
```

```python
# performance do modelo no treino

treinamento.history.keys()
plt.plot(treinamento.history['val_loss']) # evolução do erro azul

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
```

<div align = center> <img align src = /img/erro.png> </div>
<strong><p align = center> ERRO</p></strong>

<div align = center> <img align src = /img/confusao.png> </div>
<strong><p align = center> MATRIZ DE CONFUSAO</p></strong>


<h4 align = 'center'>SUBMETENDO O MODELO AOS DADOS DE TESTE</h4>

Os dados de teste tem a mesma estrutura dos dados de treino, com a única diferença de não conter as informações a respeito da classe (Survived). Essa informação deverá ser extraída pelo nosso modelo utilizando as informações dadas pelos atributos dos dados de teste.

Eis o cabeçalho dos dados de teste:


<div align = center> <img align src = /img/testhead.png> </div>

- Primeiro o arquivo contendo os dados de teste é lido, e, assim como nos dados de treino, um tratamento é feito nos valores faltantes identificados:

```python
caminhoteste = "./dados/test.csv"

test = pd.read_csv(caminhoteste)

test['Age'].fillna(mediana, inplace=True)
test[['Cabin']] = moda.transform(test[['Cabin']])
test['Embarked'].fillna('S', inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)

test.info()
```

O método "info()" mostra que o tratamento ocorreu de forma correta:

<div align = center> <img align src = /img/infotest.png> </div>

Note que nenhum atributo acusa valores faltantes (NAN).

- O pré-processamento continua fazendo os mesmos tratamentos realizados no conjunto de treino, já que os dados de teste devem estar no mesmo formato dos dados de treino quando foram submetidos ao modelo no treinamento:

```python
labeltest = LabelEncoder()

test['Sex'] = labeltest.fit_transform(test['Sex'])
test['Embarked'] = labeltest.fit_transform(test['Embarked'])

atributosteste = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

atributosteste = atributosteste.values

atributosteste = tf.convert_to_tensor(atributosteste, dtype=tf.float32)

atributosteste = normalizador.fit_transform(atributosteste)
```

- Por fim, os dados de teste são submetidos ao modelo:

```python
previsaoTeste = rede.predict(atributosteste)

previsaoTeste = (previsaoTeste > 0.55)
```
Como a previsão é feita em termos de probabilidade, é verificado se o valor da probabilidade prevista está acima de 55%, assim, é estabelecido uma métrica para avaliar se a previsão será análisada como 0 (False) ou 1 (True).

<h2 align = center>📈 Resultados</h2>

Nesta parte, os resultados são apresentados no formato de dataset, trazendo os atributos 'PassengerID' e 'Name' do conjunto 'test.csv' juntamente com o resultado da previsao, adicionada como um atributo chamado 'Survived'. O dataset de resultado é gerado na pasta contendo o arquivo .py e se chama 'Resultado.csv'.

```python
previsaoTeste = previsaoTeste.flatten()
salvar = pd.DataFrame({'PassengerID': test['PassengerId'], 'Name': test['Name'], 'Survived': previsaoTeste})
salvar.to_csv('Resultado.csv', index=False)
```
O método ".flatten()" é usado para transformar o array contendo as previsões em um array unidimensional, para que ele possa ser adicionado ao dataset no formato de coluna.

Eis o cabeçalho do arquivo 'Resultados.csv':


<div align = center> <img align src = /img/resulthead.png> </div>

<h2 align = center>🔧 Compilação e execução </h2>

Para compilação e execução correta do código, a versão recomendada do Python é a 3.8.x ou superiores.

Também é importante ressaltar que as bibliotecas devem estar instaladas corretamente no ambiente de execução, caso contrário, erros vão ocorrer.