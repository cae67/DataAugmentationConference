from functools import partial
# import keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, concatenate, SpatialDropout1D, TimeDistributed, Layer, AlphaDropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
# from keras import backend as K
from sklearn.model_selection import GroupShuffleSplit
from functools import partial
# from keras.callbacks import *
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score


import sklearn
from sklearn.metrics import confusion_matrix

# General Libraries
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftfreq, ifft
import h5py
import os

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

folderpath = '/data/users2/cellis42/Spectral_Explainability/PreTraining/Data/'
filepath = [folderpath + 'segmented_hc1_data_v2.npy',
            folderpath + 'segmented_hc2_data_v2.npy',
            folderpath + 'segmented_mdd1_data_v2.npy',
            folderpath + 'segmented_mdd2_data_v2.npy']

for i in np.arange(4):

    f = np.load(filepath[i],allow_pickle=True).item()
    
    if i == 0:
        data = f['data']
        labels = f['label']
        groups = f['subject']
    else:
        data = np.concatenate((data,f['data']),axis=0)
        labels = np.concatenate((labels,f['label']),axis=0)
        groups = np.concatenate((groups,f['subject']),axis=0)
        channels = f['channels']
                
channels2 = []
for i in range(19):
    channels2.append(channels[i].strip('EEG ').strip('-L'))

channels = channels2
channels2 = []

data = np.swapaxes(data,1,2)


## Define MDD Model

def get_model():  
        
    dropout1= 0.1
    dropout2= 0.4

    n_timesteps = 5000
    n_features = 19

    convLayer = partial(Conv1D,activation='elu',kernel_initializer='he_normal',padding='valid',
                        kernel_constraint=max_norm(max_value = 1))

    model = Sequential()

    kernel_size = 20

    model.add(convLayer(filters = 15, 
                        kernel_size= kernel_size, 
                        strides=1, 
                        input_shape=(n_timesteps, n_features), 
                        data_format='channels_last'))
    model.add(MaxPooling1D(pool_size=2, strides=2))
    model.add(BatchNormalization())

    conv_units = np.array([15, 15, 20])
    for block in range(0,3): 
        model.add(convLayer(filters = conv_units[block], 
                            kernel_size= kernel_size, 
                            strides=1))
        model.add(MaxPooling1D(pool_size=2, strides=2))
        model.add(BatchNormalization())

    model.add(Flatten())
    model.add(AlphaDropout(rate= dropout1))

    for dense_block in range(2):
        model.add(Dense(units = 16, activation='elu', kernel_initializer='he_normal', kernel_constraint=max_norm(max_value = 1),name = f"dense_l{dense_block}"))
        model.add(AlphaDropout(rate= dropout2))
   
    model.add(Dense(2, activation='softmax', kernel_initializer='glorot_normal', kernel_constraint=max_norm(max_value = 1),name="dense_output"))

    return model

batch_norm_layer_idx = [2, 5, 8, 11]
  
##################################################################################################################################################################################

# Spectral Explainability Function
def Perturbation_Freq(model,X,Y):
    
    N = np.shape(X)[0]
    Fs = 200 # Sampling Rate
    timestep = 1/Fs # Step Size
    
    # Define Frequency Bins
    bins = [];
    bins.append([0,4]) # delta
    bins.append([4,8]) # theta
    bins.append([8,12]) # alpha
    bins.append([12,25]) # beta
    bins.append([25,45]) # gamma 1
    bins.append([55,75]) # gamma 2
    bins.append([75,100]) # gamma 3


    bins = np.array(bins)
    
    n_bins = np.shape(bins)[0]
    
    initial_pred = np.argmax(model.predict(X, batch_size=128),axis=1)
    
    acc_1 = accuracy_score(Y,initial_pred)
    
    freq = np.fft.fftfreq(np.shape(X)[1], d=timestep) # 5000 sample frequencies
    
    # Identify Frequency Values Associated with Each Frequency Bin
    bins2 = np.zeros_like(freq) # preallocate array to store marker that identifies bin
    
    for bin_val in range(np.shape(bins)[0]): # for each frequency band
        positive = np.logical_and(freq>bins[bin_val,0]*np.ones_like(freq),freq<bins[bin_val,1]*np.ones_like(freq)) # indices between positive frequencies
        negative = np.logical_and(freq<-1*bins[bin_val,0]*np.ones_like(freq),freq>-1*bins[bin_val,1]*np.ones_like(freq)) # indices between negative frequencies
        vals = positive + negative # all samples within bin (OR the arrays)
        bins2[vals] = bin_val*np.ones((np.sum(vals),)) # assign marker to frequency values in each bin
    
    # Perturbation Explainability
    
    acc_change = np.zeros((n_bins,))
    
    #perform fft for all channels
    fft_vals = np.fft.fft(X_test,axis=1)
    
    for bin_val in range(n_bins): # iterate over each frequency band
        
        # Duplicate Samples
        fft_vals2 = fft_vals.copy()
        
        # Zero-out Frequency Values
        fft_sub = np.zeros_like(fft_vals2[:,np.squeeze(list(bins2 == bin_val*np.ones_like(bins2))),:])
        fft_vals2[:,np.squeeze(list(bins2 == bin_val*np.ones_like(bins2)))] = fft_sub

        # Convert Perturbed Samples Back to Time Domain
        feature_ifft = np.fft.ifft(fft_vals2,axis=1);
        X_2 = feature_ifft

        after_pred = np.argmax(model.predict(X_2, batch_size=128),axis=1)
        acc_2 = accuracy_score(Y,after_pred)

        acc_change[bin_val] = 100*(acc_2 - acc_1)/acc_1

        print('Freq ' + str(bin_val))
                        
    return (acc_change)

###################################################################################################################################

# Spatial Perturbation Function

def Perturbation_Channel(model,X,Y):
    
    N = np.shape(X)[0] 
    N_Timepoints = np.shape(X)[1]
    N_Channels = np.shape(X)[2]
    
    initial_pred = np.argmax(model.predict(X, batch_size=128),axis=1)
    
    acc_1 = accuracy_score(Y,initial_pred)
    
    # Perturbation Explainability
    
    acc_change = np.zeros((19,1))
    
    for channel in np.arange(N_Channels):
            X_2 = X.copy()
            
            # Replace channel with zeros
            X_2[:,:,channel] = np.zeros((N,N_Timepoints))

            after_pred = np.argmax(model.predict(X_2, batch_size=128),axis=1)
            acc_2 = accuracy_score(Y,after_pred)

            acc_change[channel] = 100*(acc_2 - acc_1)/acc_1
            
            print('Channel ' + str(channel))
                        
    return (acc_change)

######################################################################################################################################################
# Explainability Analysis

# Models 1.1 through 7.2
index = ['1_v1','1_v2','1_v3','2','2_v2','3','4','4_v2','5','5_v2','6','6_v2','7','7_v2']

for md in range(14):
    string = index[md]
    
    save_path = '/data/users2/cellis42/Spectral_Explainability/DataAugmentation/EMBC2024/'
    
    spectral_importance = []; spatial_importance = [];
    fold = 0

    gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3) # 1
    for tv_idx, test_idx in gss.split(data, labels, groups):

        file_path = save_path + "Models/model_m" + string  + "_fold"+str(fold)+".hdf5"
        

        clear_session()
        X_test = data[test_idx]
        y_test = labels[test_idx]

        # Model Importance

        clear_session()
        model = get_model()
        model.load_weights(file_path)
        learning_rate = 0.001 # not necessarily correct value, just placeholder
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                     metrics = ['accuracy'])

        spectral_importance.append(Perturbation_Freq(model,X_test,y_test))
        spatial_importance.append(Perturbation_Channel(model,X_test,y_test))
        fold += 1

    results_filename = save_path + 'Importance/Model_m' + string + "_importance.mat"
    savemat(results_filename, {"spectral_importance":spectral_importance,"spatial_importance":spatial_importance})
