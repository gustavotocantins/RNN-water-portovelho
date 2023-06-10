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


def salvar_informacao(epoca,neuronio,taxa, rmse):
    try:
        df = pd.read_excel('resultados.xlsx')
    except:
        df = pd.DataFrame()

    new_data = pd.DataFrame({
        'epocas': [epoca],
        'neuronio': [neuronio],
        'taxa': [taxa],
        'RMSE':[rmse]
    })

    df = pd.concat([df, new_data], ignore_index=True)

    df.to_excel('resultados.xlsx', index=False)   

def salvar_modelo(nome):
    filename = fr'C:\Users\gusta\Desktop\Laboratorio\Rede Atualizada\Rede Neural Recorrente\modelos\{nome}.pkl'
    with open(filename, 'wb') as file:  
        pickle.dump(model, file)

#=============== CRIAÇÃO DO MODELO DA REDE ===============================

# Define o conjunto de hiperparâmetros para teste

model = Sequential()
model.add(LSTM(units=128, input_shape=(9,1), return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compilador
optim = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])

#Treinamento
model.fit(enttreino, saida_des_treino, epochs=1000, batch_size=32, verbose=True)

# Resultados
teste = model.predict(entteste)
from sklearn.metrics import mean_squared_error 
eqm = mean_squared_error(saida_teste,teste) 
ermse = np.sqrt(eqm)
print('o RMSE é: ',ermse, f"| Neuro: {18}, epocas: 1000, taxa de aprendizagem: 0.01")

#Verificar se o modelo é bom
if ermse < result:
    #salvar modelo
    print("Salvando modelo...")
    result = ermse
    salvar_modelo(f"rmse-{ermse}-n{neuronio}-e{1000}-t{0.01}")
else:
    pass
salvar_informacao(1000,neuronio,0.01,ermse)

input("FINALIZADO")