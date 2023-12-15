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

## Define Base Model with Hyperparameters from M1.1

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

## Model 1.3: Train Baseline Model with Triplicate Training Data

# Define Tuner
class CVTuner(kt.Hyperband):
    def run_trial(self, trial, x, y, groups, batch_size=128, epochs=10):
        
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
            
            # Duplicate Data
            X_train = np.vstack((X_train, X_train, X_train))
            y_train = np.hstack((y_train, y_train, y_train))
            
            # Create Weights for Model Classes
            class_weights = get_class_weights(y_train)
            
            # Define Callbacks
            callbacks=[EarlyStopping(monitor='val_accuracy',patience=5)]
            
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
        
        return val_accuracy

###################################################################################################################################

# Create Tuner 
tuner = CVTuner(
    hypermodel=MyHyperModel(),
    objective = kt.Objective("val_accuracy", direction="max"),
    executions_per_trial=1, # number of times each trial is initialized (b/c different initializations get different results)
    max_epochs=40, #40
    overwrite=True,
    seed=0,
    directory=save_path,
    project_name="model_m1_v3_logs")

# Run Tuner
tuner.search(data, labels, groups, epochs=10) # 10

# Print Best HyperParameters
best_hps= tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps) 
print(best_hps.values)

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
# Train Optimal Model M1.3

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

    # Duplicate Data
    X_train = np.vstack((X_train, X_train, X_train))
    y_train = np.hstack((y_train, y_train, y_train))
    
    # Build the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)        

    # Define Path for Saving Models and Callbacks
    save_model_path = save_path + "Models/model_m1_v3_fold"+str(i)+".hdf5"
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
results_filename = save_path + "Performance/model_m1_v3.mat"
savemat(results_filename,{"validation_metrics":validation_metrics,"testing_metrics":testing_metrics})

