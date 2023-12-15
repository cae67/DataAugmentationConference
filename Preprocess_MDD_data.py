# -*- coding: utf-8 -*-
"""
Created on Wed Sept  7 16:32 2022

@author: antho
"""

#%% Import Libraries
import mne
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.io import savemat

#%% Link to Dataset

# https://figshare.com/articles/dataset/EEG_Data_New/4244171

#%% Load, Normalize, and Segment Data (Assumes that HC and MDD Data are in Separate Directories)

folderpath1 = "C:/Users/antho/Documents/Calhoun_Lab/Projects/Spectral_Explainability/MDD/RawData/HC/"
folderpath2 = "C:/Users/antho/Documents/Calhoun_Lab/Projects/Spectral_Explainability/MDD/RawData/MDD/"

target_fs = 200

for f in range(0,2):
    if f == 0:
        folderpath = folderpath1
    else:
        folderpath = folderpath2
        
    # Find Files
    files = os.listdir(folderpath)
    
    subject = []
    data = []
    for i in range(0,len(files)):
        # Load Files with Downsampling
        file=mne.io.read_raw_edf(os.path.join(folderpath,files[i])).resample(sfreq=target_fs)
        data_len = np.shape(file.times)[0]
        nepochs = np.int(np.floor(data_len/(target_fs*25))-1)
        events = mne.make_fixed_length_events(file, start=0, stop=nepochs*25-1, duration=2.5) # make events every 2.5 seconds
        nepochs = np.shape(events)[0]
        # epoch_file =  mne.Epochs(file, events, tmin=0, tmax=5,baseline=None)
        epoch_file =  mne.Epochs(file, events, tmin=-12.5, tmax=12.5,baseline=None) # When making epochs, define them as 12.5 seconds before to 12.5 seconds after previously defined events
        file = []
        
        # Get Channels of Interest
        df=epoch_file.to_data_frame()
        epoch_file = []
        channels = df.columns[3:]
        channels_to_use = ['EEG Fp1-LE', 'EEG Fp2-LE', 'EEG F7-LE', 'EEG F3-LE', 'EEG Fz-LE', 'EEG F4-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG C3-LE', 'EEG Cz-LE', 'EEG C4-LE', 'EEG T4-LE', 'EEG T5-LE', 'EEG P3-LE', 'EEG Pz-LE', 'EEG P4-LE', 'EEG T6-LE', 'EEG O1-LE', 'EEG O2-LE']
        
        epochs = df.epoch
        df=df[channels_to_use]
    
        # Z-Score Each Channel
        mean_vals = df.mean(axis=0)
        sd_vals = df.std(axis=0)
        for ch in channels_to_use:
            df[ch] -= mean_vals[ch]
            df[ch] /= sd_vals[ch]
        
        # Define Subject IDs
        if f == 0:
            subj = int(int(files[i][3:5])+f*100) # NC subject numbers are their actual numbers, MDD start from 100 + their subject number
        else:
            subj = int(int(files[i][5:7])+f*100) # NC subject numbers are their actual numbers, MDD start from 100 + their subject number
        
        # Iteratively Select and Store Epochs
        count = 0
        for epoch in range(np.min(epochs),np.max(epochs)+1):
            
            # Select Data from Epoch
            vals = np.array(df.iloc[list(np.arange(len(epochs))[list(epoch*np.ones_like(epochs)==epochs)])]).transpose()[None]
            # Remove Any Extra Points at End of Epoch
            vals = vals[...,:25*target_fs]
            
            # Store Epoch in Data Array
            if i + epoch == np.min(epochs):
                data = list(vals)
            else:
                data = np.append(data,vals,axis=0)
            count+=1
        
        df = []
        
        # Make List of Subject IDs
        if i == 0:
            subject = list((subj*np.ones((count,))).astype(int))
        else:
            subject.extend(list((subj*np.ones((count,))).astype(int)))
        
        print(i)
    
    # Create Labels and Define File Names 
    if f == 0:
        
        label = np.zeros_like(subject)
        filename1 = 'C:/Users/antho/Documents/Calhoun_Lab/Projects/Spectral_Explainability/MDD/segmented_hc1_data_v2'
        filename2 = 'C:/Users/antho/Documents/Calhoun_Lab/Projects/Spectral_Explainability/MDD/segmented_hc2_data_v2'

    else:
        
        label = np.ones_like(subject)
        filename1 = 'C:/Users/antho/Documents/Calhoun_Lab/Projects/Spectral_Explainability/MDD/segmented_mdd1_data_v2'
        filename2 = 'C:/Users/antho/Documents/Calhoun_Lab/Projects/Spectral_Explainability/MDD/segmented_mdd2_data_v2'
    
    # Split Data and Save in Separate Files
    n_samples_per_file = np.int(np.floor(len(label)/2))
    
    save_data1 = {'data':data[:n_samples_per_file,...],'subject':subject[:n_samples_per_file],'channels':channels_to_use,'label':label[:n_samples_per_file]}

    np.save(filename1,save_data1)
    
    save_data1 = [];
    
    save_data2 = {'data':data[n_samples_per_file:,...],'subject':subject[n_samples_per_file:],'channels':channels_to_use,'label':label[n_samples_per_file:]}

    np.save(filename2,save_data2)
    
    save_data2 = [];