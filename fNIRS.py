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
path_raw=os.path.join(path_data, "NIRS_Raw_DATA")
#%%initiate variables
files= glob.glob(os.path.join(path_raw, "*"))


#%%boucle principale
for i,file in enumerate(files):
    raw= mne.io.read_raw_nirx(file)
    sub_id= file.split("_")[-1]
    raw.plot(duration=5)
#%%on ouvre les données

#%%on convertit l'intensité en densité optique

#%%correction des artefacts

#%%tansformation en concentration


#%%Filtre


#%%analyse

