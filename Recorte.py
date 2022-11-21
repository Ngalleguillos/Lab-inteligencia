# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:57:11 2022

@author: Niko
"""
import os
import os.path
from obspy import read, read_inventory
import pandas as pd
from obspy.signal.trigger import classic_sta_lta,trigger_onset
from Funciones import Funciones_features
from obspy.core import UTCDateTime

def Recorte(SAC,CTM,nro_del_evento,t_extra):
    fs_real = SAC[0].stats.sampling_rate
    sac_res = SAC.copy()
    sac_k = sac_res.copy()
    fs = int(sac_k[0].stats.sampling_rate)
    cantidad_sismos = CTM['Inicio'].size #Cantidad de sismos detectados
    #Definir tiempo inicial y final del corte
    if cantidad_sismos>1:
        tin_deteccion = int(CTM['Inicio'][nro_del_evento]*fs  + t_extra*fs)
        tfin_deteccion = int(CTM['Inicio'][nro_del_evento]*fs + CTM['Duracion'][nro_del_evento]*fs)
    else:
        tin_deteccion = int(CTM['Inicio']*fs  + t_extra*fs)
        tfin_deteccion = int(CTM['Inicio']*fs + CTM['Duracion']*fs)

    #Recorte
    sac_k[0] = sac_res[0].slice(UTCDateTime(sac_res[0].stats.starttime + tin_deteccion/fs), UTCDateTime(sac_res[0].stats.starttime + tfin_deteccion/fs))
    return sac_k


def Picar_P(traza):
    fs=traza[0].stats['sampling_rate']
    sac_filt_s= Funciones_features.butter_bandpass_lfilter(traza[0].data, lowcut=1, highcut= 2, fs=fs, order=3)
    cft = classic_sta_lta(sac_filt_s, int(9 * fs), int(32 * fs))
    try:
        frame_p =trigger_onset(cft, 1.8, 0.5)[0][0]
    except:
        print('fallo sta/lta y por lo que se fijo que la P se ubica al inicio de la traza')
        frame_p = 0
    return frame_p



if False:
    from obspy.signal.trigger import plot_trigger
    traza =sac_corte
    fs=traza[0].stats['sampling_rate']
    sac_filt_s= Funciones_features.butter_bandpass_lfilter(traza[0].data, lowcut=1, highcut= 2, fs=fs, order=3)
    cft = classic_sta_lta(sac_filt_s, int(9 * fs), int(32 * fs))
    
    traza[0].data = sac_filt_s
    
    plot_trigger(traza[0], cft, 1.8, 0.5)
  
    
    plt.plot()    
    plt.plot(sac_filt_s)
    
    frame_p =trigger_onset(cft, 1.8, 0.5)[0][0]












