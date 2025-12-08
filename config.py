# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:58:03 2025

@author: gauta
"""

#%%packages
import os

#%%#Path for each person with access to the code
rootpath_alice="C://Users//gauta//OneDrive//Documents//GitHub//fNIRS_projetM2"

rootpath_marina="C:/Users/marin/Documents/GitHub/fNIRS_projetM2"

rootpath_imen ="C:/Users/imen4/Documents/fNIRS_projetM2"

#add the path to the Projet_fNIRS file


if 'gauta' in os.getcwd():
    rootpath = rootpath_alice
elif 'imen4' in os.getcwd():
    rootpath = rootpath_imen

elif 'marin' in os.getcwd():
    rootpath = rootpath_marina
#to complete for each new user

#elif 'name' in os.getcwd() :
    #rootpath = rootpath_name
