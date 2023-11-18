import numpy as np
#from datetime import datetime as dt
#from datetime import timedelta
import matplotlib.pyplot as plt



def moving_average(vector, filt, amplitude,opts='same'):
    '''moving average filter
    la convoluzione per un vector di n pesi uguali (vector i cui elementi sommano 1)
    restituisce la media mobile su finestre larghe 'amplitude' (si può ottenere una media mobile pesata con valori arbitrari dei pesi)
    '''

    if filt==None:
        if amplitude==None:
            amplitude=3
        filt = 1/amplitude*np.ones(amplitude)
    V = np.convolve(vector,filt,opts)
    rd_len = int(np.size(V,axis=0))   # reduced length
    return V, rd_len
