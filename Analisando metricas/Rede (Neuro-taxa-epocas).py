#Importar bibliotecas
from time import time
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle 
#Importar bibliotecas
from time import time
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle 
import pandas as pd
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
t0 = time()

xx = pd.read_excel(r"T1T2T3T4T5T6PRPVPRCJPRFZNAPV.xlsx",engine='openpyxl')
x = xx.copy()
grupo = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'PRPV','PRCJ','PRFZ', 'NAPV']

for column in grupo: 
    x[column] = x[column]  / x[column].max()

ent = np.array(x[['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'PRPV','PRCJ','PRFZ']],dtype='float32')
namostra, nvariavel = ent.shape 
nteste = int(0.1*namostra) + 1 
ntreino = namostra - nteste 
enttreino = ent[0:ntreino,:] 
entteste = ent[ntreino:namostra,:] 
saida_des = np.array(x['NAPV'],dtype='float32') 
saida_des_treino = saida_des[0:ntreino] 
saida_teste = saida_des[ntreino:namostra] 


def salvar_modelo(nome):
    filename = fr'C:\Users\gusta\Desktop\Laboratorio\Rede Atualizada\Analisando metricas\modelos\{nome}.pkl'
    with open(filename, 'wb') as file:  
        pickle.dump(model, file)

#=============== CRIAÇÃO DO MODELO DA REDE ===============================

# Define o conjunto de hiperparâmetros para teste
neuronios_conj = [10, 20, 40, 60,100, 200, 500, 1000]


# Loop para testar cada combinação de hiperparâmetros

for neuronio in neuronios_conj:
    #Modelo da rede
    seed = 42  # Substitua 42 pelo valor de seed que você deseja utilizar
    tf.random.set_seed(seed)
    model = tf.keras.Sequential() 
    model.add(tf.keras.layers.Dense(neuronio,input_shape=(9,),
                activation="sigmoid",
                kernel_initializer='zeros'))
    model.add(tf.keras.layers.Dense(1,kernel_initializer='zeros'))


    # Compilador
    optim = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])

    #Treinamento
    model.fit(enttreino, saida_des_treino, epochs=1000, batch_size=32, verbose=False)

    teste1 = model.predict(entteste) #10% dos dados
    teste2 = model.predict(ent) #Geral

    from sklearn.metrics import mean_squared_error 
    def calcular_rmse(real,predicao):
        eqm = mean_squared_error(real,predicao) 
        ermse = np.sqrt(eqm)
        return ermse

    #Fazer predição para os dados de 10%
    print("Quantidade de neuronios: ", neuronio)
    print('RMSE de 10% dos dados: ',calcular_rmse(saida_teste,teste1))
    print('RMSE de toda a serie: ',calcular_rmse(saida_des,teste2))

    print()
    #salvar_modelo(f"n{neuronio}-e{1000}-t{0.01}")

input("FINALIZADO")
