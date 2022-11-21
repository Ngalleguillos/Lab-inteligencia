# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:51:21 2022

@author: Niko
"""
import numpy as np
import tensorflow as tf
from obspy.signal.filter import bandpass
from obspy.signal.trigger import classic_sta_lta, trigger_onset
import os
from obspy import read, read_inventory
import math
import time
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from keras_multi_head import MultiHeadAttention
from obspy.core import UTCDateTime
import csv

import fx
import funcion_BAZ
import Recorte
import test_magnitude_estimation
from estimacion_epicentro import estimacion_epicentro
import warnings
warnings.filterwarnings("ignore")

path_root =  os.getcwd()

dir_sac = os.path.join(path_root+'\data\sac\\').replace('\\', '/');   # sac_E2 es la otra opcion
dir_ctm = os.path.join(path_root,'data/','Viterbi_DNN_test'+'.ctm').replace('\\', '/');   # Viterbi_mono_asociacion_E2.ctm es la otra opcion

#Listas para ejes
lista_sac = os.listdir(dir_sac)

#Leer CTM
data_ctm = pd.read_csv(dir_ctm, sep=" ", names=['Evento','Nro','Inicio','Duracion','Tipo_de_Deteccion']) #Se lee el ctm
data_ctm.set_index('Evento', inplace=True)

t_extra = -5 #segundos
cant_sismos = data_ctm['Inicio'].size #Cantidad de sismos detectados
eje_sliced=[]
salidas=[]

control_ejes=[]
salidaDict = {'Distancia': [], 'Latitud':[], 'Longitud':[], 'BAZ':[], 'Magnitud': [],
          'Estacion':[], 'TiempoInicio':[], 'TiempoFin':[]}
#Loop recorre toda la señal
for i in range(cant_sismos):
    print('-------------------------------')
    starttime = time.time()

    for file in lista_sac:

        #Recorte
        sac = read(dir_sac+file) #Se lee el SAC
        eje=sac[0].stats['channel'][-1]
        sac_corte=Recorte.Recorte(sac,data_ctm,i,t_extra)

        #Encontrar P
        if eje == 'Z':
            print('Sismo {:d}-esimo, a las: {}'.format(i,str(sac_corte[0].stats.starttime)))  
            P_pick=Recorte.Picar_P(sac_corte)

        #Generar entradas para modulos de Distancia y Back Azimuth

        if eje == 'Z':
            sac_Z=sac_corte.copy()
        elif eje=='N':
            sac_N=sac_corte.copy()
        elif eje=='E':
            sac_E=sac_corte.copy()

        # if eje_sliced==[]:
        #     sac_conc=sac_corte.copy()
        # else:
        #     sac_conc+=sac_corte.copy()

        eje_sliced.append(eje)

        #Si se pasa por los tres ejes, se calculan la Distancia y BAZ
        if ('Z' in eje_sliced) and ('N' in eje_sliced) and ('E' in eje_sliced):
            sac_conc=sac_Z.copy()
            sac_conc+=sac_E.copy()
            sac_conc+=sac_N.copy()

            magnitude = test_magnitude_estimation.magnitude_estimation(sac_conc)
            print('Magnitud estimada de: ', magnitude)

            #Ver como solucionar el tema de la P en 0
            if P_pick==0:

                P_pick = int(len(sac_conc[0].data)/2)
                #eje_sliced=[]
                #continue



            BAZ=funcion_BAZ.calculate_BAZ(sac_conc,P_pick, detectar_p = False, rad = False)
            dist=fx.distance(sac_conc,p=P_pick,p_calcular=False)
            
            
            print('BAZ estimada de: ', BAZ)
            print('Distancia estimada de: ', dist)

            lat, lon = estimacion_epicentro(sac[0].stats['station'], dist, BAZ)
            print('Epicentro| latidud: {} y longitud: {} '.format(lat, lon))
            print("--- %s segundos por sismo ---" % int((time.time() - starttime)))
            
            
            # salidas.append([dist,float(BAZ)])
            salidas.append([dist, lat, lon, magnitude, sac[0].stats['station'], sac_Z[0].stats['starttime'], sac_Z[0].stats['endtime']  ])
           
            salidaDict['Distancia'].append(dist)
            salidaDict['Latitud'].append(lat)
            salidaDict['Longitud'].append(lon)
            salidaDict['BAZ'].append(BAZ)
            salidaDict['Magnitud'].append(magnitude)
            salidaDict['Estacion'].append(str(sac[0].stats['station']))            
            salidaDict['TiempoInicio'].append(str(sac[0].stats['starttime']))            
            salidaDict['TiempoFin'].append(str(sac[0].stats['endtime']))            





            # control_ejes.append(eje_sliced)
            eje_sliced=[]
            

salidaDict = pd.DataFrame(salidaDict)

salidaDict.to_csv('reporte.csv',index=False)