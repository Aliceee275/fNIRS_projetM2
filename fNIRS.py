# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:49:04 2025

@author: gauta
"""

#%%import packages
import mne
import os
import config as cfg
import glob
import matplotlib
#%%create paths
path_root=cfg.rootpath
path_data=os.path.join(path_root, "NIRS_DATA")
path_raw=os.path.join(path_data, "NIRS_Raw_Data")
#%%initiate variables
files= glob.glob(os.path.join(path_raw, "*"))


#%%boucle principale
for i,file in enumerate(files):
    #on ouvre le fichier
    raw= mne.io.read_raw_nirx(file)
    sub_id= file.split("_")[-1]
    raw.plot(duration=5)
#%%on renomme les anotations
    new_annotations= {'Easy': '2.0', 'Hard': '1.0'}
    events,event_ids= mne.events_from_annotations(raw)
    raw.annotations.rename({'1.0': 'Hard', '2.0':'Easy'})
    event_ids=new_annotations

#%%on convertit l'intensité en densité optique
    raw_od= mne.preprocessing.nirs.optical_density(raw)
    raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)
    #mdofi
    #sous-échantillonage à 2Hz
    sfreq= raw_od.info['sfreq']
    raw_od=raw_od.resample(2)
    raw_od.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)
    
#%%correction des artefacts (TDDR)
    raw_tddr= mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)
    raw_tddr.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)
    
#%%tansformation en concentration
    raw_c= mne.preprocessing.nirs.beer_lambert_law(raw_tddr)
    raw_c.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)
    
#%%Filtre
    raw_filtered= raw_c.filter(0.02,0.3)
    raw_filtered.plot(n_channels=len(raw_od.ch_names), duration=500, show_scrollbars=False)
    
#%%analyse
    #event_ids=new_annotations
    epochs=mne.Epochs(
        raw_filtered, 
        events,
        event_id= [1,2], 
        tmin = -5,
        tmax = 30,
        baseline= None,
        preload=True,
        reject= None
        )
    
