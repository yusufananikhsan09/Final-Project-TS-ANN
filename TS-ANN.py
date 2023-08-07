import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Sequential
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import math
import glob

# =================================================================================
# Input Data dan Proses Normalisasi 

## Data yang digunakan dalam format .npy
# df = np.load(r'D:\STUDI\Tugas Akhir\DATASET\PengolahanML\NEWDATA_BGT\MLDisp_3.npy')
# tahun_latih = 5
# jumlah_epoch = 30
# tm_ac = tahun_latih*jumlah_epoch
# numdata = 4216

### Melakukan transformasi dimensi dari gambar menjadi berurutan berdasarkan indeks raster
#data = np.zeros((tahun_latih*jumlah_epoch*numdata,1))
#for k in range(tm_ac):
#    for j in range(62):
#        for i in range(68):
#            if j>0:
#                data[i+j+k*numdata+j*67,] = df[k,j,i]
#            else:
#                data[i+j+k*numdata,] = df[k,j,i] 

## Data yang digunakan adalah format csv
df = pd.read_csv(r"D:\STUDI\Tugas Akhir\DATASET\PengolahanML\NEWDATA_BGT\MergeML_v3.csv")
df['Disp']= pd.to_numeric(df.Disp, errors='coerce')
data = data.Disp.copy
## min dan max data
mindata = min(data)
maxdata = max(data)

## Normalisasi
m = len(data)
dataNorm = np.zeros((m,1))
for i in range(m):
    dataNorm[i] = (data[i] - mindata)/(maxdata-mindata)

# ==================================================================================
# Membagi Data Pelatihan dan Pengujian
arr = np.zeros((numdata,tm_ac,1))
data_latih_norm = np.zeros((numdata, jumlah_epoch*tahun_latih-(2*jumlah_epoch), jumlah_epoch))
target_latih_norm = np.zeros((numdata, jumlah_epoch*tahun_latih-2*jumlah_epoch, 1))
data_prediksi_norm = np.zeros((numdata,jumlah_epoch*tahun_latih-2*jumlah_epoch, jumlah_epoch))
predicted = np.zeros((numdata, 3*jumlah_epoch,1))

## Melakukan Pembagian data menjadi data pelatihan dan pengujian
## Penyusunan data dilakukan secara windowing dan berdasarkan indeks piksel
## arr[indeks_piksel, timeseries]
for i in range(numdata):
    for j in range(tm_ac):
        if i>0:
            arr[i,j,] = data[j*numdata+i]
        else:
            arr[i,j,] = data[j*numdata]
            
for i in range(numdata):
    for m in range(jumlah_epoch*tahun_latih-(2*jumlah_epoch)):
        for n in range(jumlah_epoch):
            data_latih_norm[i,m,n] = arr[i,m+n,]
            
for i in range(numdata):
    for m in range(jumlah_epoch*tahun_latih-2*jumlah_epoch):
        for n in range(jumlah_epoch):
            target_latih_norm[i,m,] = arr[i,jumlah_epoch+m]

for i in range(numdata):
    for m in range(jumlah_epoch*tahun_latih-2*jumlah_epoch):
        for n in range(jumlah_epoch):
            data_prediksi_norm[i,m,n] = arr[i,2*jumlah_epoch+m,]
        
# ==============================================================================            
# Membuat Model dan Prediksi
def predict_disp(TRAIN, TRAIN_TARGET, PREDICT, EPOCH):
    tf.keras.backend.clear_session()
    inpShape = TRAIN.shape[1]
    model = Sequential([Dense(150, input_shape=[inpShape], activation = 'relu'),
                        Dense(100, activation='relu'),
                        Dense(50, activation='relu'),
                        Dense(30, activation='relu'),
                        Dense(10, activation='relu'),
                        Dense(1)
                        ])
    model.compile(optimizer =tf.keras.optimizers.Adam(1e-8),
                  loss='mse',
                  metrics=['mse'])
    model.fit(TRAIN, TRAIN_TARGET, epochs=int(EPOCH), verbose=2)
    return model.predict(PREDICT)
    

for i in range(numdata):
    predicted[i,:,:] = predict_disp(data_latih_norm[i,:,:], 
                                    target_latih_norm[i,:,:], 
                                    data_prediksi_norm[i,:,:], 
                                    100)
    predicted[i,:,:] = predicted[i,:,:]*(maxdata-mindata)+mindata