# Deep Learning Libraries

from functools import partial
# import keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, GlobalMaxPool1D, GlobalAveragePooling1D, Dense, Dropout, BatchNormalization, Activation, concatenate, SpatialDropout1D, TimeDistributed, Layer, AlphaDropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
import keras_tuner as kt

import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from sklearn.model_selection import GroupShuffleSplit
from functools import partial
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import accuracy_score, recall_score, balanced_accuracy_score, f1_score, precision_score

import sklearn
from sklearn.metrics import confusion_matrix

# General Libraries
from scipy.io import loadmat, savemat
from scipy.fft import fft, fftfreq, ifft
import h5py
import os

# Check GPU Availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.config.list_physical_devices('GPU')

# Define Directory Path for Saving Results
save_path = '/data/users2/cellis42/Spectral_Explainability/DataAugmentation/EMBC2024/'

# Define Filepaths for Loading Data
folderpath = '/data/users2/cellis42/Spectral_Explainability/PreTraining/Data/'
filepath = [folderpath + 'segmented_hc1_data_v2.npy',
            folderpath + 'segmented_hc2_data_v2.npy',
            folderpath + 'segmented_mdd1_data_v2.npy',
            folderpath + 'segmented_mdd2_data_v2.npy']

# Load Training Data
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

# Get Channel Names
channels2 = []
for i in range(19):
    channels2.append(channels[i].strip('EEG ').strip('-L'))

channels = channels2
channels2 = []

# Rearrange Data Axes
data = np.swapaxes(data,1,2)

## Define Base Model

class MyHyperModel(kt.HyperModel):
    
    def build(self,hp):
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

        learning_rate = hp.Choice("lr",[1e-3,7e-4,5e-4,3e-4,1e-4,7e-5,5e-5,3e-5,1e-5])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                         metrics = ['accuracy'])

        return model

batch_norm_layer_idx = [2, 5, 8, 11]

# Smooth Time Mask Augmentation

def my_sigmoid(t):
    
    return 1 / (1 + np.exp(-t))
        
def smooth_time_augmentation(X_train,delta_t,lambda_val):
    # X_train - data in shape n_samples x n_timepoints x n_channels
    # delta_t - number of time points to mask
    # lambda_val - temperature: steepness of sigmoids
    
    X_train = X_train.copy()
    
    n_samples = np.shape(X_train)[0]
    n_channels = np.shape(X_train)[2]
    
    t_max = np.shape(X_train)[1]
    t = np.arange(1,t_max+1)
        
    for i in range(n_samples):
        
        np.random.seed(i)
        
        t_cut = np.round(np.random.uniform(low = 1, high = (t_max - delta_t)))
        # print(t_cut)
        
        term1 = -1*(t - t_cut)
        term2 = t - t_cut - delta_t
        mask = my_sigmoid(lambda_val * term1) + my_sigmoid(lambda_val * term2)
                
        X_train[i,...] = X_train[i,...] * np.repeat(np.expand_dims(mask,axis=1),n_channels,axis=1)
    
    return X_train
    
# Compute Class Weights

def get_class_weights(y_train):
    values, counts = np.unique(y_train, return_counts=True)

    weights = class_weight.compute_class_weight(class_weight = 'balanced', classes = np.unique(y_train), y = np.squeeze(y_train))
    class_weights = dict(zip(values, weights))
    
    return class_weights

# Split Data for Cross-Validation

def split_data(x,y,groups,tv_idx,test_idx,gss_train_val):
   
    # Split Data for Outer Fold
    X_train_val, X_test = x[tv_idx], x[test_idx]
    y_train_val, y_test = y[tv_idx], y[test_idx]
    group = groups[tv_idx]
    
    # Split Data for Inner Fold
    for train_idx, val_idx in gss_train_val.split(X_train_val, y_train_val, group):

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    
    return(X_train,y_train,X_val,y_val,X_test,y_test)

## Model 4.1: Model Trained with 1 Copy of Smooth Time Augmented Data

# Define Tuner
class CVTuner(kt.Hyperband):
    def run_trial(self, trial, x, y, groups, save_path, count, batch_size=128, epochs=10,augmentation_param=0.3):
        
        val_accuracy = []
        
        # Define Training, Validation, and Test Splits
        gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state=3) #10
        gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.78, random_state = 3)
        
        for tv_idx, test_idx in gss.split(x, y, groups):
            
            # Get Training, Validation, and Test Sets
            X_train,y_train,X_val,y_val,X_test,y_test = split_data(x,y,groups,
                                                                   tv_idx,test_idx,
                                                                   gss_train_val)
            
            # Get KerasTuner HyperParameters
            hp = trial.hyperparameters
            
            # Perform Data Augmentation
            X_train = np.vstack((X_train, smooth_time_augmentation(X_train,augmentation_param,0.5)))
            y_train = np.hstack((y_train, y_train))
            
            # Create Weights for Model Classes
            class_weights = get_class_weights(y_train)
            
            # Define Callbacks
            callbacks=[EarlyStopping(monitor='val_accuracy',patience=10)]
            
            # Build and Train Model
            model = self.hypermodel.build(hp)
            model.fit(X_train, to_categorical(y_train), validation_data=(X_val, to_categorical(y_val)),
                      shuffle=True, verbose=0, batch_size=batch_size, epochs=epochs, 
                      class_weight=class_weights,callbacks=callbacks)
            
            # Compute Validation Performance
            val_accuracy.append(model.evaluate(X_val, to_categorical(y_val))[1])
        
        # Print Trial ID
        print(trial.trial_id)
        
        # Compute Mean Performance Across Folds
        val_accuracy = np.mean(val_accuracy)
        
        # Save Performance from Trial if Better Than Previous Trials for Data Augmentation Parameter Value
        val_accuracy2 = np.squeeze(loadmat(save_path)['val_accuracy'])
        val_accuracy2[count] = np.max([val_accuracy2[count],val_accuracy])
        savemat(save_path, {'val_accuracy':val_accuracy2})
        
        return val_accuracy

###################################################################################################################################

# Initialize List for Storing Optimal Hyperparameters
hps = []

# Smooth Time Mask Parameters Values
fs = 200 # sampling rate in Hertz
param_vals = np.arange(1,10)/5*fs # number of time points between 0 and 2 seconds

# Initialize File for Saving Scores
save_path_val_logs = save_path +'val_scores_m4.mat'
savemat(save_path_val_logs, {'val_accuracy':np.zeros_like(param_vals)})

# Iteratively Train for Different Augmentation Parameters
count = 0
for param in param_vals:
    # Create Tuner
    tuner = CVTuner(
        hypermodel=MyHyperModel(),
        objective = kt.Objective("val_accuracy", direction="max"),
        executions_per_trial=1, # number of times each trial is initialized (b/c different initializations get different results)
        max_epochs=40, #40
        overwrite=True,
        seed=0,
        directory=save_path,
        project_name="model_m4_logs")
    
    # Run Tuner
    tuner.search(data, labels, groups, save_path_val_logs, count, epochs=10, augmentation_param = param) # 10

    # Print Best HyperParameters
    best_hps= tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hps) 
    print(best_hps.values)
    hps.append(best_hps)
    count += 1

# Get Best Hyperparameters
top_param_idx = np.argmax(list(np.squeeze(loadmat(save_path_val_logs)['val_accuracy'])))
best_hps= hps[top_param_idx]
best_param = np.array(param_vals)[top_param_idx]

# Compute Performance Metrics
def compute_metrics(y,y_pred):
    
    acc = accuracy_score(y, y_pred)
    sens = recall_score(y, y_pred, pos_label=1)
    spec = recall_score(y, y_pred, pos_label=0)
    bacc = balanced_accuracy_score(y, y_pred)
    
    return [acc,sens,spec,bacc]

# Print Metrics
def print_metrics(metrics):
    
    # Convert List of Metrics to Array
    metrics = np.array(metrics)
    
    # Print Metrics
    print(metrics)
    
    # Print Mean and SD of Metrics in Table
    print(pd.DataFrame(data=[metrics.mean(axis=0), metrics.std(axis=0)], 
                       index=['mean','std'], columns=['acc','sens','spec','bacc']))
    
################################################################################################################
# Train Optimal Model 4.1

tf.random.set_seed(41) # best is seed 42, v7

testing_metrics = []; validation_metrics = [];

i = 0

# Define Training, Validation, and Test Splits
gss = GroupShuffleSplit(n_splits = 25, train_size = 0.9, random_state = 3) # 10
gss_train_val = GroupShuffleSplit(n_splits = 1, train_size = 0.78, random_state = 3)

for tv_idx, test_idx in gss.split(data, labels, groups):
    
    print(i)
    # Get Training, Validation, and Test Sets
    X_train,y_train,X_val,y_val,X_test,y_test = split_data(data,labels,groups,
                                                           tv_idx,test_idx,
                                                           gss_train_val)
        
    # Perform Data Augmentation
    X_train = np.vstack((X_train, smooth_time_augmentation(X_train,best_param,0.5)))
    y_train = np.hstack((y_train, y_train))

    # Build the model with the optimal hyperparameters
    best_hps= hps[top_param_idx]
    model = tuner.hypermodel.build(best_hps)        

    # Define Path for Saving Models and Callbacks
    save_model_path = save_path + "Models/model_m4_fold"+str(i)+".hdf5"
    early_stopping = EarlyStopping(monitor="val_accuracy",patience=10)
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=True)

    # Create Weights for Model Classes
    class_weights = get_class_weights(y_train)

    # Train Model
    history = model.fit(X_train, to_categorical(y_train), epochs= 40, batch_size = 128, validation_data=(X_val, to_categorical(y_val)), 
                        shuffle=True, verbose = 0, callbacks=[checkpoint,early_stopping],class_weight=class_weights)

    # Load Best Model for Evaluation
    model.load_weights(save_model_path)

    # Compute Test Performance Metrics
    preds = np.argmax(model.predict(X_test, batch_size=128),axis=1)
    testing_metrics.append(compute_metrics(y_test,preds))
    
    # Compute Validation Performance Metrics
    preds_val = np.argmax(model.predict(X_val, batch_size=128),axis=1)
    validation_metrics.append(compute_metrics(y_val,preds_val))
    
    i += 1

# Visualize Validation and Test Performance
print("Validation Set Metrics")
print_metrics(validation_metrics)

print("Test Set Metrics")
print_metrics(testing_metrics)

# Save Validation and Test Performance
results_filename = save_path + "Performance/model_m4.mat"
savemat(results_filename,{"validation_metrics":validation_metrics,"testing_metrics":testing_metrics})

