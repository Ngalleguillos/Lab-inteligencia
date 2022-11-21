# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 09:11:11 2021

@author: Marc
"""




def red():


    from keras.models import Sequential, Model
    from keras.layers import Dense, LSTM, Bidirectional,Concatenate,Input,concatenate
    import numpy as np
    from tensorflow.keras.utils import Sequence
    import time
    from Funciones import Funciones_entrenamiento
    from tensorflow.python.keras import backend as K
    import tensorflow as tf
    from keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    import os
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    import joblib


    path_root =  os.getcwd()
 #   param_i=pd.read_csv(os.path.join(path_root, 'param_i.csv').replace('\\', '/'), delimiter=',',index_col=0,squeeze=True)
 #   tipo_red=param_i['red']
 #   val_pat=float(param_i['patience'])
 #   lr=float(param_i['lear_rate'])


    # adjust values to your needs
    # config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 1} )
    # config.gpu_options.allow_growth = True
    # sess = tf.compat.v1.Session(config=config)

    # K.set_session(sess)




    '''

    np.random.seed(123)
    start_time = time.time()

    path_feat_in_train_lstm = os.path.join(path_root,'Features/feat_lstm_raw_train_locacion.npy').replace('\\', '/')
    path_feat_in_train_mlp = os.path.join(path_root,'Features/feat_mlp_raw_train_locacion.npy').replace('\\', '/')
    path_loc_real_train = os.path.join(path_root,'Features/locacion_raw_train.npy').replace('\\', '/')

    path_feat_in_val_lstm = os.path.join(path_root,'Features/feat_lstm_raw_val_locacion.npy').replace('\\', '/')
    path_feat_in_val_mlp = os.path.join(path_root,'Features/feat_mlp_raw_val_locacion.npy').replace('\\', '/')
    path_loc_real_val = os.path.join(path_root,'Features/locacion_raw_val.npy').replace('\\', '/')
    path_feat_in_test_lstm = os.path.join(path_root,'Features/feat_lstm_raw_test_locacion.npy').replace('\\', '/')
    path_feat_in_test_mlp = os.path.join(path_root,'Features/feat_mlp_raw_test_locacion.npy').replace('\\', '/')
    path_loc_real_test = os.path.join(path_root,'Features/locacion_raw_test.npy').replace('\\', '/')


    '''

    class MyBatchGenerator_lstm_mlp(Sequence):
        'Generates data for Keras'
        def __init__(self, X, x, y, batch_size=1, shuffle=False):
            'Initialization'
            self.X = X
            self.x = x
            self.y = y
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.on_epoch_end()

        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.y)/self.batch_size))

        def __getitem__(self, index):
            return self.__data_generation(index)

        def on_epoch_end(self):
            'Shuffles indexes after each epoch'
            self.indexes = np.arange(len(self.y))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, index):
            Xb = np.empty((self.batch_size, *self.X[index].shape))
            xb = np.empty((self.batch_size, *self.x[index].shape))
            yb = np.empty((self.batch_size, *self.y[index].shape))
            # naively use the same sample over and over again
         #   for s in range(0, self.batch_size):
            Xb = self.X
            xb = self.x
            yb = self.y

            return [Xb, xb], yb



    target = 'Distancia' #'Coordenadas', 'Distancia'
    tipo_de_escalamiento_features = 'MinMax' #'Standard', 'MinMax', 'MVN', 'None'
    repeats = 3  #Nro de veces que se correran modelos de estimacion
    tipo_de_escalamiento_target = 'Standard'


    model = tf.keras.models.load_model('bueno.h5')


    path_feat_in_x_lstm = os.path.join(path_root,'Features/feat_lstm_raw_x_locacion.npy').replace('\\', '/')
    path_feat_in_x_mlp = os.path.join(path_root,'Features/feat_mlp_raw_x_locacion.npy').replace('\\', '/')


    feat_in_x_lstm = np.load(path_feat_in_x_lstm, allow_pickle=True)
    feat_in_x_mlp = np.load(path_feat_in_x_mlp, allow_pickle=True)

    min_f_train_lstm = np.load('norm_min_lstm.npy', allow_pickle=True)
    max_f_train_lstm = np.load('norm_max_lstm.npy', allow_pickle=True)
    min_f_train_mlp = np.load('norm_min_mlp.npy', allow_pickle=True)
    max_f_train_mlp = np.load('norm_max_mlp.npy', allow_pickle=True)
    my_scaler = joblib.load('scaler.gz')

    feat_norm_x_lstm = np.array([(feat_in_x_lstm[i]-min_f_train_lstm)/(max_f_train_lstm-min_f_train_lstm)
                                    for i in range(len(feat_in_x_lstm))],dtype=object)


    feat_norm_x_mlp = np.array([(feat_in_x_mlp[i]-min_f_train_mlp)/(max_f_train_mlp-min_f_train_mlp)
                                    for i in range(len(feat_in_x_mlp))],dtype=object)
    temporal_feat=np.array(feat_norm_x_lstm)
    global_feat=np.array(feat_norm_x_mlp)

  #  x_test =    MyBatchGenerator_lstm_mlp(temporal_feat,global_feat, np.zeros((3,1)), batch_size=1, shuffle=False)
  #  x_test =    feat_norm_x_lstm

   # y_estimada_test = model.predict(x_test)
    y_estimada_test = model.predict([tf.convert_to_tensor(temporal_feat.astype('float32')),tf.convert_to_tensor(global_feat.astype('float32'))],verbose=0)

    #print(y_estimada_test[0])

    preds = my_scaler.inverse_transform(y_estimada_test)
    return float(preds[0])





def caract(sac_k,p,p_calcular=False):
    #Si no hay P picada, y quiere calcularse automáticamente, no agregar p o igualar a 0
    import numpy.matlib
    import os
    import os.path
    import numpy as np
    from obspy import read, read_inventory
    import math
    import time
    from Funciones import Funciones_features
    from Funciones import Funciones_ufro_uach
    import scipy as sp
    import pandas as pd
    from obspy.signal.trigger import classic_sta_lta,trigger_onset
    import json
    from sklearn.preprocessing import OneHotEncoder
    start_time = time.time()
    path_root =  os.getcwd()
    umbral_corte = 1 # 0.03  #0.1 es un 10% # al fijarlo en 1 no se corta la traza
    frame_length = 4 #segundos
    frame_shift = 2  #segundos
    Vel = False
    Energy='E3'
    escala_features_fft = 'logaritmica' # 'lineal' o 'logaritmica'   #FFT de la traza
    escala_features_energia = 'logaritmica' # 'lineal' o 'logaritmica'  Energia del frame
    modo_id_estacion='contador'

    caract_glob = {'Envolvente subida': 1,'Envolvente bajada': 1 ,'S-P':1,'Coordenadas estaciones':0,'Azimuth':0}


    id_estacion = {'PB09':1,'PB06':2,'AC02':3,'CO02':4,'PB14':5,'CO01':6,'GO01':7,'GO03':8, 'MT16':9, 'PB18':10, 'PATCX':11,'PB02':12,
                  'MNMCX':13,'PB01':14,'PB03':15,'PB05':16,'PB07':17,'PB04':18,'PB15':19,'PB11':20,'PSGCX':21,'TA01':22,'VA03':23,
                  'PB12':24,'HMBCX':25,'MT03':26,'AP01':27,'AC05':28,'MT02':29,'TA02':30,'GO04':31,'PB10':32,'MT05':33,'AC04':34,
                  'CO03':35,'GO05':36,'VA01':37,'PB16':38,'AC01':39,'CO05':40,'CO06':41,'GO02':42,'VA06':43,'PX06':44,
                  'CO04':45,'BO03':46,'AC07':47,'PB08':48,'CO10':49}



    # if sac_k[0].stats.channel[-1] =='Z':
        #magnitud_por_evento.append([lista_eventos[i],float(sac_k[0].stats.sac.mag)])
        #magnitud_por_evento.append(mag)
        #if (Base[0:3]!='Gra' and Base[0:1]!='M'):
        # lat,lon, depth,dist = sac_k[0].stats.sac.evla,sac_k[0].stats.sac.evlo,sac_k[0].stats.sac.evdp, sac_k[0].stats.sac.user0
     #   locacion_por_evento['Latitud'].append(lat)
     #   locacion_por_evento['Longitud'].append(lon)
     #   locacion_por_evento['Profundidad'].append(depth)
            # locacion_por_evento['Distancia'].append(dist)
            # locacion_por_evento['Evento'].append(lista_eventos[i])
        #     magnitud_por_evento.append([lista_eventos[i],float(sac_k[0].stats.sac.mag)])
        # else:
        #     magnitud_por_evento.append(mag)

    #Las señales que estan en 100Hz pasan a 40 Hz
    fs = int(sac_k[0].stats.sampling_rate)
    fs_real = int(sac_k[0].stats.sampling_rate)
    if fs ==100:
        sac_k[0].data = sp.signal.resample(sac_k[0].data,int(len(sac_k[0].data)*40/100))
        sac_k[0].stats.sampling_rate = 40
        sac_k = sac_k.slice(sac_k[0].stats.starttime+1, sac_k[0].stats.endtime-1)
        fs = 40
    elif fs==40:
        sac_k = sac_k.slice(sac_k[0].stats.starttime+1, sac_k[0].stats.endtime-1)
    elif fs!=40:
        print('Hay sampling rate distinto a 40, revisar!!')


    frame_len = frame_length*fs
    frame_shi = frame_shift*fs
    nfft = pow(2,math.ceil(np.log2(frame_len)))

    """
    if Vel == True: # CONVERTIR A VELOCIDAD LAS SEÑALES
        sta = sac_k[0].stats.station
        cha = sac_k[0].stats.channel
        if sta[0:2] == 'PB':
            inv = read_inventory('xml/'+sta[0:2]+'.xml')
            sac_k.remove_response(inventory=inv, output="VEL")
        else:
            inv = read_inventory('xml/'+sta+'/'+cha+'.xml')
            sac_k.remove_response(inventory=inv, output="VEL")
            """

    if Vel == True: # CONVERTIR A VELOCIDAD LAS SEÑALES
        sta = sac_k[0].stats.station
        cha = sac_k[0].stats.channel


        try:
            #if sta[0:2]=='PB':
            #    inv = read_inventory('xml/'+sta[0:2]+'.xml')
            #    sac_k.remove_response(inventory=inv, output="VEL")
            #elif sta in ['PB09','PB06','AC02','CO02','PB14','CO01','GO01','GO03', 'MT16', 'PB18']:
            #    inv = read_inventory('xml/'+sta+'/'+cha+'.xml')
            #    sac_k.remove_response(inventory=inv, output="VEL")

            #else:
            inv = read_inventory('nuevos_xml/'+sta+'.xml')

            sac_k.remove_response(inventory=inv, output="VEL")
            remover_resp = True
        except:
            print('Problema con el .xml, no se puede remover resp. instrumental')
            remover_resp = False


      #  data_k = Funciones_features.butter_highpass_lfilter(sac_k[0].data, cutoff=1, fs=fs, order=3) #Filtro pasa alto causal
    data_k = Funciones_features.butter_highpass_lfilter(sac_k[0].data, cutoff=1, fs=fs, order=3) #Filtro pasa alto causal




    if sac_k[0].stats.channel[-1] =='Z': #Verifico que estoy en el canal Z para el calculo del tiempo s-p y corte de coda
        sac_filt_s= Funciones_features.butter_bandpass_lfilter(sac_k[0].data, lowcut=1, highcut= 2, fs=fs, order=3)

        if False:
              import matplotlib.pyplot as plt
              plt.plot(sac_filt_s)

        if p_calcular:

            cft = classic_sta_lta(sac_filt_s, int(9 * fs), int(32 * fs))
            try:
                frame_p =trigger_onset(cft, 1.8, 0.5)[0][0]
            except:
                print('falló sta/lta, por lo que se fijó la P al inicio de la traza')
                frame_p = 0
        else:
            frame_p=p

        frame_s = np.argmax(sac_filt_s)


        time_sp = (frame_s-frame_p)/fs

        # if math.isnan(time_sp):
        #     es_leido = 0
        #     continue


        if False:
            plt.plot(sac_filt_s)
            plt.plot(frame_p,0,'y*')
            plt.plot(frame_s,0,'r*')
            plt.title( conjunto+str(i)+'  sp='+str(time_sp)+'   dist='+str(dist))
         #   plt.title( lista_eventos[i]+'    '+conjunto+str(i))
         #   plt.show()
            plt.savefig('C:/Users/Desktop 0/Documents/SISMOS/LSTM/Figures/'+lista_eventos[i]+'.png')
            plt.close()



        ##Corte de Coda. En el caso de usar 3 canales la coda se corta en el mismo punto en los 3 canales, siempre se tomo como referencia el canal Z
        #Tipo de energia de referencia con que se corta la coda
        if Energy == 'E1':
            Energia_Z_ref = Funciones_features.E1(data_k, frame_len, frame_shi, nfft,escala = 'lineal')
        elif Energy == 'E2':
            Energia_Z_ref = Funciones_features.E2(data_k, frame_len, frame_shi,escala = 'lineal')
        elif Energy == 'E3':
            Energia_Z_ref = Funciones_features.E3(data_k, frame_len,frame_shi,escala = 'lineal')
        if umbral_corte ==1:
            muestra_corte_coda = len(data_k)
            energia_umbral_corte = np.max(Energia_Z_ref)
            feat_Energy_or = Funciones_features.E3(data_k, frame_len,frame_shi)#esto solo lo uso para el ploteo de mas abajo
            data_k_or = data_k
        else:
            arg_amp_maxima = np.argmax(Energia_Z_ref) #supuesto: la energia maxima esta en la onda S
            arg_amp_minima = np.argmin(Energia_Z_ref[:arg_amp_maxima]) # tomo la energia minima entre el inicio y la S
            delta_energia = Energia_Z_ref[arg_amp_maxima]-Energia_Z_ref[arg_amp_minima] # Diferencia de energia entre la minima entre el inicio y la s y la energia de la onda S
            energia_umbral_corte = delta_energia*umbral_corte+Energia_Z_ref[arg_amp_minima]

            arg_fin_nueva_coda = arg_amp_maxima + np.argmin(np.abs(Energia_Z_ref[arg_amp_maxima:]-energia_umbral_corte))
            muestra_corte_coda = int(fs*frame_len*arg_fin_nueva_coda/frame_shi)
            data_k_or = data_k
            feat_Energy_or = Funciones_features.E3(data_k, frame_len,frame_shi)#esto solo lo uso para el ploteo de mas abajo


    # if math.isnan(time_sp):
    #     es_leido = 0
    #     continue


       # data_k = data_k[:muestra_corte_coda] #A la traza se le corta la coda

        #A la traza enventanadada se le obtiene abs(FFT//2), puede ir con o sin logaritmo
    feat_k = Funciones_features.parametrizador(data_k, frame_len, frame_shi,nfft, escala = escala_features_fft)

    #Distintos tipos de energia por ventana
    if Energy == 'E1':
        feat_Energy = Funciones_features.E1(data_k, frame_len, frame_shi, nfft,escala = escala_features_energia)
        feat_k = np.hstack((feat_k, np.array([feat_Energy]).T ))
    elif Energy == 'E2':
        feat_Energy = Funciones_features.E2(data_k, frame_len, frame_shi,escala = escala_features_energia)
        feat_k = np.hstack((feat_k, np.array([feat_Energy]).T ))
    elif Energy == 'E3':
        feat_Energy = Funciones_features.E3(data_k, frame_len,frame_shi,escala = escala_features_energia)
        feat_k = np.hstack((feat_k, np.array([feat_Energy]).T))


    #Features temporales de la traza de un canal
    features_canales_temporal=(feat_k)
    feat_canales_temporal = (features_canales_temporal)


    envC = Funciones_ufro_uach.up_level_idx(data_k, 200, 50, 10)/fs
    envC_ = Funciones_ufro_uach.down_level_idx(data_k, 200, 50, 10)/fs

    #Features globales por canal
    if sac_k[0].stats.channel[-1] =='Z':
     #Built-in mutable seq

        #features_canales_global.append(np.hstack(([id_estacion[estaciones[i]],envC,envC_])))
        if modo_id_estacion=='contador':
            features_canales_global=[id_estacion[sac_k[0].stats.sac.kstnm]]
        else:
            features_canales_global=id_estacion==estaciones[i]

        if caract_glob['Envolvente subida']==1:
            features_canales_global.append(envC)
        if caract_glob['Envolvente bajada']==1:
            features_canales_global.append(envC_)
        if caract_glob['S-P']==1:
            features_canales_global.append(time_sp)
        if caract_glob['Coordenadas estaciones']==1:
            features_canales_global.append(Coordenadas_estaciones[estaciones[i]])
        if caract_glob['Azimuth']==1:
            features_canales_global.append(az)


    #features_canales_global.append(np.hstack(([id_estacion[estaciones[i]],time_sp,envC,envC_])))

        #features_canales_global.append(np.hstack(([id_estacion[estaciones[i]],Coordenadas_estaciones[estaciones[i]],envC,envC_])))
        #features_canales_global.append(np.hstack(([id_estacion[estaciones[i]],Coordenadas_estaciones[estaciones[i]],time_sp])))

    elif sac_k[0].stats.channel[-1] =='N' or sac_k[0].stats.channel[-1] =='E' :
        features_canales_global.append(np.hstack(([envC, envC_]))) #Variables en canales N y E


    feat_canales_global =  np.hstack(features_canales_global)


    feat_por_evento_global=[]#variables que van a la MLP
    feat_por_evento_temporal=[]#variables que van a la LSTM

    #locacion_por_evento2.append(locacion_por_evento['Evento'][i])

    #Todas las features del evento

    for i in range(3):
        feat_por_evento_global.append(feat_canales_global)#variables que van a la MLP
        feat_por_evento_temporal.append(feat_canales_temporal)#variables que van a la LSTM


    # feat_por_evento_global=(feat_canales_global)#variables que van a la MLP
    # feat_por_evento_temporal=(feat_canales_temporal)#variables que van a la LSTM



    # ---------------------------------------------------------------------------------------------------------------



    # try:
    #     dim_feat_lstm = feat_canales_temporal.shape[0],feat_canales_temporal.shape[1]
    #     print('Dimension features LSTM: {:d}x{:d}'.format(dim_feat_lstm))
    # except Exception as e:
    #  #   print(f"DEBUG: {e}")
    #     continue



    # ---------------------------------------------------------------------------------------------------------------

    # print('Dimension features LSTM: {:d}x{:d}'.format(features_canales_temporal.shape[0],features_canales_temporal.shape[1]))

    # print('Dimension features MLP: {:d}'.format(feat_canales_global.shape[0]))

    # print('***********************')
    # print('Se leyeron {} de un total de asd eventos'.format(len(feat_por_evento_temporal)))




    # #Se aletorizan los eventos
    # ind_random = np.random.permutation(np.arange(0,len(feat_por_evento_temporal)))
    # feat_por_evento_temporal = np.array(feat_por_evento_temporal, dtype=object)[ind_random]
    # feat_por_evento_global = np.array(feat_por_evento_global, dtype=float)[ind_random]
    # #magnitud_por_evento = np.array(magnitud_por_evento)[ind_random]
    # locacion_por_evento['Distancia']= list(np.array(locacion_por_evento['Distancia'])[ind_random])
    # #locacion_por_evento['Longitud']= list(np.array(locacion_por_evento['Longitud'])[ind_random])
    # #locacion_por_evento['Latitud']= list(np.array(locacion_por_evento['Latitud'])[ind_random])
    # #locacion_por_evento['Profundidad']= list(np.array(locacion_por_evento['Profundidad'])[ind_random])
    # locacion_por_evento['Evento']= list(np.array(locacion_por_evento['Evento'])[ind_random])

    feat_por_evento_temporal = np.array(feat_por_evento_temporal, dtype=object)

    feat_out = os.path.join('Features/').replace('\\', '/')
    #print('Se guardaron features y target de Locacion')
    np.save(feat_out+'feat_lstm_raw_x_locacion.npy', feat_por_evento_temporal)
    np.save(feat_out+'feat_mlp_raw_x_locacion.npy', feat_por_evento_global)
  #  np.save(feat_out+'locacion_raw_x.npy', locacion_por_evento)
    # with open(path_root+'\Modelos\config_features_distancia.json' , 'w') as fp:
    #     json.dump(params, fp)

    #print("--- %s seconds ---" % int((time.time() - start_time)))



def distance(sac_k,p=0,p_calcular=False):
    caract(sac_k,p,p_calcular)
    d=red()
    return d