# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 11:42:24 2025

@author: marin
"""


import mne
import os
import config as cfg
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
from mne.viz import plot_compare_evokeds
from pprint import pprint
import pandas as pd
from copy import deepcopy
import seaborn as sns

def individual_analysis(file_path):

    #on ouvre le fichier
    raw = mne.io.read_raw_nirx(file_path, verbose=False)

    #renomme les triggers
    raw.annotations.rename({'1.0': 'Hard', '2.0': 'Easy'})

    #on convertit l'intensité en densité optique
    raw_od = mne.preprocessing.nirs.optical_density(raw)

    #sous-échantillonage à 2Hz
    raw_od = raw_od.resample(2)

    #correction des artefacts (TDDR)
    raw_tddr = mne.preprocessing.nirs.temporal_derivative_distribution_repair(raw_od)

    #tansformation en concentration
    raw_c = mne.preprocessing.nirs.beer_lambert_law(raw_tddr)

    #filtre Bandpass
    raw_filtre = raw_c.filter(0.02, 0.3, verbose=False)

    #Applique le nouveau noms aux données filtrées etc.
    event_id = {'Hard': 1, 'Easy': 2}
    events, event_ids = mne.events_from_annotations(raw_filtre, event_id=event_id)

    #Analyse 
    epochs = mne.Epochs(
        raw_filtre,
        events,
        event_id=event_id,
        tmin=-5,
        tmax=30,
        baseline=None,
        preload=True
    )

    return raw_filtre, epochs

#%% fait sur tous les participants

path_root=cfg.rootpath
path_data=os.path.join(path_root, "NIRS_DATA")
path_raw=os.path.join(path_data, "NIRS_Raw_Data")
files= glob.glob(os.path.join(path_raw, "*"))

groupe_evokeds = defaultdict(list) #permet de memoriser les triggers
for file in files:
    print("\n>>> Traitement de:", file)
    raw_filtre, epochs = individual_analysis(file)

    # Ajouter les evoked de ce participant au groupe
for condition in epochs.event_id:
    groupe_evokeds[condition].append(epochs[condition].average())

#%% visualiser le décours temporel

#1 ligne avec 2 colonnes définit par groupe_evokes
fig, axes = plt.subplots(nrows=1, ncols=len(groupe_evokeds), figsize=(17,5))
lims = dict(hbo=[-5, 12], hbr=[-5, 12])

for pick, color in zip(["hbo", "hbr"], ["r", "b"]):
    for idx, evoked in enumerate(groupe_evokeds):
        plot_compare_evokeds(
            {evoked: groupe_evokeds[evoked]},
            combine="mean", 
            picks=pick,
            axes=axes[idx], #evoked
            show=False,
            colors=[color],
            legend=False,
            ylim=lims,
            ci=0.95,  #interval de confiance 
            show_sensors=idx == 2,
        )
        axes[idx].set_title(f"{evoked}")
axes[0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])

#%% sur les régions d'intéret 

# Specify channel pairs for each ROI
left = [[1,1], [2,1], [2,2], [3,2]]
center = [[3, 3], [5,4], [4, 4]]
right = [[5,5], [6,6], [7,6], [8,7]]

def picks_pair_to_idx(raw, pairs):
    idxs = []
    for s, d in pairs:
        for ch_idx, ch_name in enumerate(raw.info['ch_names']):
            if f"S{s}_D{d}" in ch_name:
                idxs.append(ch_idx)
    return idxs

# Then generate the correct indices for each pair and store in dictionary
rois = dict(
    Left_Hemisphere=picks_pair_to_idx(raw_filtre, left),
    Center=picks_pair_to_idx(raw_filtre, center),
    Right_Hemisphere=picks_pair_to_idx(raw_filtre, right),
)

pprint(rois)

#%% graph par rois 

# Specify the figure size and limits per chromophore.
fig, axes = plt.subplots(nrows=len(rois), ncols=len(groupe_evokeds), figsize=(15, 6))
lims = dict(hbo=[-8, 16], hbr=[-8, 16])

for pick, color in zip(["hbo", "hbr"], ["r", "b"]):
    for ridx, roi in enumerate(rois):
        for cidx, evoked in enumerate(groupe_evokeds):
            if pick == "hbr":
                picks = rois[roi][1::2]  # Select only the hbr channels
            else:
                picks = rois[roi][0::2]  # Select only the hbo channels

            plot_compare_evokeds(
                {evoked: groupe_evokeds[evoked]},
                combine="mean",
                picks=picks,
                axes=axes[ridx, cidx],
                show=False,
                colors=[color],
                legend=False,
                ylim=lims,
                ci=0.95,
                show_sensors=cidx == 2,
            )
            axes[ridx, cidx].set_title([evoked])
        axes[0, cidx].set_title(f"{evoked}")
        axes[ridx, 0].set_ylabel(f"{roi}\nChromophore (ΔμMol)")
axes[0, 0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])

#%% amplitude 6s post-stimulus 

df = pd.DataFrame(columns=["ID", "ROI", "Chroma", "Condition", "Value"])
subj_id = 0
for idx, evoked in enumerate(groupe_evokeds):
    
    for subj_data in groupe_evokeds[evoked]:
        subj_id += 1
        for roi in rois:
            for chroma in ["hbo", "hbr"]:
                data = deepcopy(subj_data).pick(picks=rois[roi]).pick(chroma)
                value = data.crop(tmin=5.0, tmax=7.0).data.max() * 1.0e6 #prend 6s & amplitude max

                # Append metadata and extracted feature to dataframe
                this_df = pd.DataFrame(
                    {
                        "ID": subj_id,
                        "ROI": roi,
                        "Chroma": chroma,
                        "Condition": evoked,
                        "Value": value,
                    },
                    index=[0],
                )
                df = pd.concat([df, this_df], ignore_index=True)
                
df.reset_index(inplace=True, drop=True)
#df["Value"] = pd.to_numeric(df["Value"])  # some Pandas have this as object



#%%



#%% visualise en graph 



#Hbo
sns.catplot(
    x="Condition",
    y="Value",
    hue="ID",
    data=df.query("Chroma == 'hbo'"),
    errorbar=None,
    palette="muted",
    height=4,
    s=10,
)

#hbr

sns.catplot(
    x="Condition",
    y="Value",
    hue="ID",
    data=df.query("Chroma == 'hbr'"),
    errorbar=None,
    palette="muted",
    height=4,
    s=10,
)
