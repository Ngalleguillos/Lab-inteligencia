#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import tensorflow as tf
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from sklearn.preprocessing import OneHotEncoder
from keras_multi_head import MultiHeadAttention
import os
from obspy import read, read_inventory
import math
import time
import scipy as sp
import pandas as pd


# In[2]:


#algunas funciones a utilizar


class VanillaPositionalEncoding(tf.keras.layers.Layer):

    def get_angles(self,pos, i, d_model):
        angles = 1 / np.power(10000., (2*(i//2)) / np.float32(d_model))
        return pos * angles # (seq_length, d_model)

    def PositionalEncoding(self,inputs_layer):
        # input shape batch_size, seq_length, d_model
        seq_length = inputs_layer.shape.as_list()[-2]
        d_model = inputs_layer.shape.as_list()[-1]
        # Calculate the angles given the input
        angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                                 np.arange(d_model)[np.newaxis, :],
                                 d_model)
        # Calculate the positional encodings
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        # Expand the encodings with a new dimension
        pos_encoding = angles[np.newaxis, ...]
        #plt.pcolormesh(pos_encoding[0], cmap='viridis')
        return inputs_layer + tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return self.PositionalEncoding(inputs)

#sirve para normalizar por el valor mas extremo
def sec_div_max(secuencia):
    #secuencia en formato samples,timestep,features.
    for i in range(len(secuencia)):
        maximo = np.max([np.abs(secuencia[i].max()),np.abs(secuencia[i].min())])
        secuencia[i] = secuencia[i]/maximo
    return secuencia

def to_angle(seno,coseno, corregir = True):
    ### corregir pasa los ángulos negativos a [pi,2pi]
    angle = np.arctan2(seno,coseno)

    if corregir == False:
        return angle
    else:
        for i in range(len(angle)):
            if angle[i]<=0:
                angle[i] = angle[i]+2*np.pi

        return angle


# In[4]:


# receives obspy seismic signal considering the channels in order Z, E, N; and the list of counts that indicate P-wave arrivals.

def calculate_BAZ(signal,P_counts, detectar_p = False, rad = True):
    #signal, P_counts, detectar_p = sac_conc, P_pick, False
    global one_hot_enc, model_BAZ

    #check if the models were loaded
    try: one_hot_enc, model_BAZ
    #if not, they are loaded as global variables
    except:

        one_hot_enc = OneHotEncoder()
        id_estacion = np.array(['PB09','PB06','AC02','CO02','PB14','CO01','GO01','GO03',
                                'MT16', 'PB18','AC04','AC05','AP01','CO03','GO04','HMBCX',
                                'MNMCX','MT02','MT03','MT05','PATCX','PB01','PB02','PB03',
                                'PB04','PB05','PB07','PB10','PB11','PB12','PB15','PSGCX',
                                'TA01','TA02','VA03','GO02','GO05','PB16','PB08','CO04',
                                'VA01','AC01','CO05','CO06','VA06','CO10','BO03','AC07','PX06']).reshape(-1,1)
        one_hot_enc.fit(id_estacion)

        model_BAZ = tf.keras.models.load_model('modelo_usar.h5',custom_objects =
                                               {'MultiHeadAttention':MultiHeadAttention,
                                                'VanillaPositionalEncoding': VanillaPositionalEncoding})


    temporal_feat, global_feat = BAZ_preprocessing(signal, P_counts, detectar_p)
    temporal_feat = temporal_feat.astype('float')
    temporal_feat = sec_div_max(temporal_feat)

    nro_features = model_BAZ.layers[0].input_shape[0][1] #Input shape temporal features
    if np.shape(temporal_feat)[1]==nro_features:
        prediction = model_BAZ.predict([tf.convert_to_tensor(temporal_feat.astype('float32')),tf.convert_to_tensor(global_feat)],verbose=0)
        prediction = to_angle(prediction[:,1],prediction[:,0])
    else:
        prediction =np.nan

    if rad == False:
        return prediction*180/np.pi
    else:
        return prediction


# In[39]:


def BAZ_preprocessing(signal, P_count, detectar_p):

#orignialmente ordenados Z,E,N
    canal_sac_Z = signal[0].copy()
    canal_sac_E = signal[1].copy()
    canal_sac_N = signal[2].copy()

    fs = int(canal_sac_Z.stats.sampling_rate)

    # se recorta la traza de manera que los 3 canales esten sincronizados (a veces no lo estan)
    inicio_maximo = max(canal_sac_Z.stats.starttime,canal_sac_E.stats.starttime,canal_sac_N.stats.starttime)
    finales = np.array([canal_sac_Z.stats.endtime,canal_sac_E.stats.endtime,canal_sac_N.stats.endtime])
    fin_minimo = np.argmin(finales)
    tiempo_inicio = canal_sac_Z.stats.starttime
    diferencia_tiempos = inicio_maximo - tiempo_inicio
    tiempo_fin = finales[fin_minimo]
    P_count-=int(diferencia_tiempos*fs)

    P_count = P_count*40/fs

    fs_real = int(canal_sac_Z.stats.sampling_rate)
    if fs==100:
        canal_sac_Z.data = sp.signal.resample(canal_sac_Z.data,int(len(canal_sac_Z.data)*40/100))
        canal_sac_Z.stats.sampling_rate = 40
        canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
        canal_sac_E.data = sp.signal.resample(canal_sac_E.data,int(len(canal_sac_E.data)*40/100))
        canal_sac_E.stats.sampling_rate = 40
        canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
        canal_sac_N.data = sp.signal.resample(canal_sac_N.data,int(len(canal_sac_N.data)*40/100))
        canal_sac_N.stats.sampling_rate = 40
        canal_sac_N = canal_sac_N.slice(inicio_maximo, tiempo_fin)
        fs = 40
    elif fs==40:
        canal_sac_Z = canal_sac_Z.slice(inicio_maximo, tiempo_fin)
        canal_sac_E = canal_sac_E.slice(inicio_maximo, tiempo_fin)
        canal_sac_N = canal_sac_N.slice(inicio_maximo, tiempo_fin)
    elif fs!=40:
        print('Hay sampling rate distinto a 40, revisar!!')



    ### remover resp instrumental
    sta = canal_sac_Z.stats.station
    cha = canal_sac_Z.stats.channel

    inv = read_inventory('nuevos_xml/'+sta+'.xml')
    canal_sac_Z.remove_response(inventory=inv, output="VEL")
    canal_sac_E.remove_response(inventory=inv, output="VEL")
    canal_sac_N.remove_response(inventory=inv, output="VEL")
    #except:
        #print('Problema con el .xml, no se puede remover resp. instrumental')
    ###


    ### Filtrado Pasabanda
    canal_sac_Z.filter('bandpass', freqmin = 0.5, freqmax = 10.0, corners=2, zerophase=True)
    canal_sac_E.filter('bandpass', freqmin = 0.5, freqmax = 10.0, corners=2, zerophase=True)
    canal_sac_N.filter('bandpass', freqmin = 0.5, freqmax = 10.0, corners=2, zerophase=True)

    if detectar_p == True:
        cft= classic_sta_lta(canal_sac_Z.data, int(3 * fs), int(6 * fs))
        P_count =trigger_onset(cft, 1.8, 0.5)[0][0] #1.8 0.5
        print('Cuenta de llegada de la P a 40[hz]: ',P_count)
    else:
        print('Cuenta de llegada de la P a 40[hz]: ',P_count)



    feat_por_evento_temporal, feat_por_evento_global = [], []

    #Modificacion para caso en que la P se pickea al inicio de la señal
    if P_count-fs*1<=0:
        pre_P=P_count
        post_p=(P_count+fs*3)
    else:
        pre_P=P_count-fs*1
        post_p=(P_count+fs*3)
    data_z = canal_sac_Z.data[int(pre_P):int(post_p)]
    data_e = canal_sac_E.data[int(pre_P):int(post_p)]
    data_n = canal_sac_N.data[int(pre_P):int(post_p)]
    #Fin modificacion

    #Lo original ^
    # data_z = canal_sac_Z.data[int(P_count-fs*1):int(P_count+fs*3)]
    # data_e = canal_sac_E.data[int(P_count-fs*1):int(P_count+fs*3)]
    # data_n = canal_sac_N.data[int(P_count-fs*1):int(P_count+fs*3)]


    feat_canales_temporal = np.squeeze(np.dstack(np.array([data_z,data_e,data_n])))
    feat_por_evento_temporal.append(feat_canales_temporal)
    ###
    ### Caracteristicas globales (solo id en formato one hot encoding)

    encoding = np.squeeze(one_hot_enc.transform(np.array(sta).reshape(-1,1)).toarray())
    feat_por_evento_global.append(encoding)

    feat_por_evento_temporal = np.array(feat_por_evento_temporal)
    feat_por_evento_global = np.array(feat_por_evento_global)

    return feat_por_evento_temporal, feat_por_evento_global

