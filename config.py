# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 10:58:03 2025

@author: gauta
"""

#%%packages
import os

#%%#chemins d'accès pour chaque personne
rootpath_alice="C://Users//gauta//OneDrive//Documents//GitHub//fNIRS_projetM2"
<<<<<<< HEAD
rootpath_marina="C:/Users/marin/Documents/GitHub/fNIRS_projetM2"
=======
rootpath_imen ="C:/Users/imen4/Documents/fNIRS_projetM2"
>>>>>>> 2ffb539378a23d06a3ae17799c1ae0d1902e0d8b
#ajouter le chemin d'accès du fichier Projet_fNIRS


if 'gauta' in os.getcwd():
    rootpath = rootpath_alice
elif 'imen4' in os.getcwd():
    rootpath = rootpath_imen

elif 'marin' in os.getcwd():
    rootpath = rootpath_marina
#compléter

#elif 'name' in os.getcwd() :
    #rootpath = rootpath_name
