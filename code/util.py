import numpy as np
#from datetime import datetime as dt
#from datetime import timedelta
import matplotlib.pyplot as plt
import csv



def moving_average(vector, filt, amplitude=3, opts='same'):
    '''moving average filter
    vector: Array
    filt: Array (filter, array of weights that add up to 1)
    amplitude: Integer (length of filt, range of the filter. Not necessary if filt is given)
    opts: String (options of the convolution; must be in ['full','valid','same'])
    
    la convoluzione per un vector di n pesi uguali (vector i cui elementi sommano a 1)
    restituisce la media mobile su finestre larghe 'amplitude' (si può ottenere una media mobile pesata con valori arbitrari dei pesi)
    '''

    if filt==None:
        filt = 1/amplitude*np.ones(amplitude)   #creation of a uniform filter
    else:
        filt_sum = sum(filt)
        if filt_sum != 1:
            for i in len(filt):
                filt[i] = filt[i]/filt_sum   #normalization of the vector
        
    V = np.convolve(vector,filt,opts)
    #rd_len = int(np.size(V,axis=0))   # reduced length
    return V   #, rd_len

def readFile(file_name):
    '''read csv file function
    file_name: String (Path of the file)
    '''
    data = []
    header = []
    with open(file_name,'r') as file:
        reader = csv.reader(file)
        #next(reader, None)
        head = True
        for row in reader:
            if head == True:
                header=[str(element) for element in row]
                head = False
            else:
                data.append([element for element in row])
    return np.array(data), header

def writeFile(file_name, data, header):   #TO BE MODIFIED
    with open(file_name,'w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(header)
        for row in data:   #TO BE MODIFIED
            writer.writerow(row)
