#Importar bibliotecas
from time import time
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle 
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
    filename = fr'C:\Users\gusta\Desktop\Laboratorio\Rede Atualizada\modelos\{nome}.pkl'
    with open(filename, 'wb') as file:  
        pickle.dump(model, file)

#=============== CRIAÇÃO DO MODELO DA REDE ===============================

# Define o conjunto de hiperparâmetros para teste
neuronios_conj = range(1,20)
result = 0.15


# Loop para testar cada combinação de hiperparâmetros

for neuronio in neuronios_conj:
    #Modelo da rede
    model = tf.keras.Sequential() 
    model.add(tf.keras.Input(9))
    model.add(tf.keras.layers.Dense(neuronio,
                activation="sigmoid",
                kernel_initializer='glorot_normal'))
    model.add(tf.keras.layers.Dense(1))

    # Compilador
    optim = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optim, loss='mse', metrics=['accuracy'])

    #Treinamento
    model.fit(enttreino, saida_des_treino, epochs=70, batch_size=2, verbose=True)

    # Resultados
    teste = model.predict(entteste)
    from sklearn.metrics import mean_squared_error 
    eqm = mean_squared_error(saida_teste,teste) 
    ermse = np.sqrt(eqm)
    print('o RMSE é: ',ermse, f"| Neuro: {neuronio}, epocas: 1000, taxa de aprendizagem: 0.01")

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
