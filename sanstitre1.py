# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 23:35:19 2025

@author: gauta
"""

#%% Import librairiesimport os
import config as cfg
import glob
import os

# Import common libraries
from collections import defaultdict
from copy import deepcopy
from itertools import compress
from pprint import pprint

# Import Plotting Library
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import StatsModels
import statsmodels.formula.api as smf
from mne import Epochs, events_from_annotations, set_log_level
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
    temporal_derivative_distribution_repair,
)

# Import MNE processing
from mne.viz import plot_compare_evokeds

# Import MNE-BIDS processing
import mne

# Import MNE-NIRS processing
from mne_nirs.channels import get_long_channels, picks_pair_to_idx
from mne_nirs.datasets import fnirs_motor_group
from mne_nirs.signal_enhancement import enhance_negative_correlation

# Set general parameters
set_log_level("WARNING")  # Don't show info, as it is repetitive for many subjects

#%%define the analysis function

def individual_analysis(path):
    # load the Nirx data
    raw_intensity= mne.io.read_raw_nirx(path, verbose=False)
    raw_intensity = get_long_channels(raw_intensity, min_dist=0.01)
    
    #rename trigger 1 and 2 into 'Hard' and 'Easy'
    #events,event_ids= mne.events_from_annotations(raw_intensity)
    raw_intensity.annotations.rename({'1.0': 'Hard', '2.0':'Easy'})

    # Convert signal to optical density and determine bad channels
    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(raw_od, h_freq=1.35, h_trans_bandwidth=0.1)
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    raw_od.interpolate_bads()

    # Down sample to 2 Hz
    raw_od.resample(2)
    
    #apply movement correction (TDDR)
    raw_od = temporal_derivative_distribution_repair(raw_od)

    # Convert to haemoglobin and filter
    raw_haemo = beer_lambert_law(raw_od, ppf=0.1)
    raw_haemo = raw_haemo.filter(
        0.02, 0.3, h_trans_bandwidth=0.1, l_trans_bandwidth=0.01, verbose=False
    )

    # Apply further data cleaning techniques and extract epochs
    raw_haemo = enhance_negative_correlation(raw_haemo)
    # Extract events but ignore those with
    # the word Ends (i.e. drop ExperimentEnds events)
    events, event_dict = events_from_annotations(
        raw_haemo
        # raw_haemo, verbose=False, regexp="^(?![Ends]).*$"
    )
    epochs = Epochs(
        raw_haemo,
        events,
        event_id=event_dict,
        tmin=-5,
        tmax=30,
        reject=dict(hbo=200e-6),
        reject_by_annotation=True,
        proj=True,
        baseline=(None, 0),
        detrend=0,
        preload=True,
        verbose=False,
    )
    

    return raw_haemo, epochs

#%% Do the analysis for each participant
#define the paths
all_evokeds = defaultdict(list)

path_root=cfg.rootpath
path_data=os.path.join(path_root, "NIRS_DATA")
path_raw=os.path.join(path_data, "NIRS_Raw_Data")
#%%initiate variables
files= glob.glob(os.path.join(path_raw, "*"))

liste_epochs=[]

for i,file in enumerate(files):
    path= file
    subject= file.split("_")[-1]
    datatype= "nirs"
    # Analyse data and return both ROI and channel results
    raw_haemo, epochs = individual_analysis(path)

    # Save individual-evoked participant data along with others in all_evokeds
    for cidx, condition in enumerate(epochs.event_id):
        all_evokeds[condition].append(epochs[condition].average())
        
print(all_evokeds)


#%%Visualisation
#visualize Hbo and Hbr for each channels for easy and hard
fig, axes = plt.subplots(nrows=1, ncols=len(all_evokeds), figsize=(17, 5))
lims = dict(hbo=[-5, 20], hbr=[-5, 20])

for pick, color in zip(["hbo", "hbr"], ["r", "b"]):
    for idx, evoked in enumerate(all_evokeds):
        plot_compare_evokeds(
            {evoked: all_evokeds[evoked]},
            #combine="mean",
            picks=pick,
            axes=axes[idx],
            show=False,
            colors=[color],
            legend=False,
            ylim=lims,
            ci=0.95,
            show_sensors=idx == 2,
        )
        axes[idx].set_title(f"{evoked}")
axes[0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])
#%% Visualisation for each ROI
#First let's define our 3 ROI: left, center and right based on the montage
left = [
    [2,1], [2,3], #source S2 
    [1,1], [1,2], #source S1 
    [3,3], [3,2], #source S3 
]

center = [
    [4,2], [4,5], #source S4 
    [5,3], [5,6], #source S5
]

right = [
    [6,5], [6,6],
    [7,5], [7,7],
    [8,6], [8,7],
]
# Then generate the correct indices for each pair and store in dictionary
rois = dict(
    Left_Hemisphere=picks_pair_to_idx(raw_haemo, left),
    Center=picks_pair_to_idx(raw_haemo, center),
    Right_Hemisphere=picks_pair_to_idx(raw_haemo, right),
)

print(rois)

#Visualisation
fig, axes = plt.subplots(nrows=len(rois), ncols=len(all_evokeds), figsize=(15, 6))
lims = dict(hbo=[-5, 20], hbr=[-5, 20])

for pick, color in zip(["hbo", "hbr"], ["r", "b"]):
    for ridx, roi in enumerate(rois):
        for cidx, evoked in enumerate(all_evokeds):
            if pick == "hbr":
                picks = rois[roi][1::2]  # Select only the hbr channels
            else:
                picks = rois[roi][0::2]  # Select only the hbo channels

            plot_compare_evokeds(
                {evoked: all_evokeds[evoked]},
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
            axes[ridx, cidx].set_title("")
        axes[0, cidx].set_title(f"{evoked}")
        axes[ridx, 0].set_ylabel(f"{roi}\nChromophore (ΔμMol)")
axes[0, 0].legend(["Oxyhaemoglobin", "Deoxyhaemoglobin"])

#%%Visualisation of maximal amplitudes for each subject

df = pd.DataFrame(columns=["ID", "ROI", "Chroma", "Condition", "Amplitude"])

for idx, evoked in enumerate(all_evokeds):
    subj_id = 0
    for subj_data in all_evokeds[evoked]:
        subj_id += 1
        for roi in rois:
            for chroma in ["hbo", "hbr"]:
                data = deepcopy(subj_data).pick(picks=rois[roi]).pick(chroma)
                value = data.crop(tmin=5.0, tmax=7.0).data.max() * 1.0e6

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
df["Value"] = pd.to_numeric(df["Value"])

#%%Visualise max amplitudes for each participant for Hbo and Hbr

for roi in rois:
    df_roi= df.loc[df['ROI']==roi]
    sns.catplot(
        x="Condition",
        y="Value",
        hue="ID",
        data=df_roi.query("Chroma == 'hbo'"),
        errorbar=None,
        palette="muted",
        height=4,
        s=10,
    )
#%% en lineplot
y_min = df.loc[df['Chroma']=='hbo', 'Value'].min()
y_max = df.loc[df['Chroma']=='hbo', 'Value'].max()+1

fig, axes = plt.subplots(nrows=1, ncols=len(rois), figsize=(17, 5))

for idx, roi in enumerate(rois):
    df_roi = df.loc[df['ROI']==roi].query("Chroma == 'hbo'")
    
    # Line plot to see the difference for a same participant
    sns.lineplot(
        x="Condition", 
        y="Value",
        hue="ID",
        data=df_roi,
        palette="muted",
        marker='o',
        ax=axes[idx],  # Utiliser l'index
        legend=True,
    )
    axes[idx].set_ylim(y_min, y_max) 
    axes[idx].set_title(f"ROI: {roi}")  # Appliquer le titre à l'axe spécifique

plt.tight_layout()  # Pour mieux espacer les subplots
plt.show()

#%%
dict_delta = {r: [] for r in rois}
for roi in df.ROI.unique():
    df_roi = df.loc[df['ROI']==roi].query("Chroma == 'hbo'")
    print(roi)
    for sub in df_roi['ID'].unique():
        easy= df_roi.loc[df_roi.ID == sub].query("Condition == 'Easy'")['Value'].values[0]
        hard= df_roi.loc[df_roi.ID == sub].query("Condition == 'Hard'")['Value'].values[0]
        print(hard)
        delta= hard-easy
        dict_delta[roi].append(delta)
        print(df_roi.Chroma)
#%%
df_delta = pd.DataFrame.from_dict(dict_delta)

#%%
plt.figure(figsize=(10, 6))

# Dégradé de violettes
pastel_colors = ["#D5E8D4",  # Vert forêt clair
                   "#66BB6A",  # Vert forêt moyen  
                   "#3D8B3D"]

sns.boxplot(data=df_delta, palette=pastel_colors)

# Stripplot avec la même palette pour éviter les doublons
sns.stripplot(data=df_delta, palette=pastel_colors, alpha=0.7, size=6, edgecolor='gray', linewidth=0.5, dodge=True)

plt.title('Boxplot with Individual Data Points')
plt.ylabel('Delta Values')
plt.show()

for col in df_delta.columns:
    print('moyenne:', df_delta[col].mean(), '+/-', df_delta[col].std())

#%% same but with effect size
import numpy as np 
from math import sqrt
dict_effect = {r: {'hard': [], 'easy': []} for r in rois}

for roi in df.ROI.unique():
    df_roi = df.loc[df['ROI']==roi].query("Chroma == 'hbo'")
    print(roi)
    for sub in df_roi['ID'].unique():
        easy= df_roi.loc[df_roi.ID == sub].query("Condition == 'Easy'")['Value'].values[0]
        hard= df_roi.loc[df_roi.ID == sub].query("Condition == 'Hard'")['Value'].values[0]
        dict_effect[roi]['easy'].append(easy)
        dict_effect[roi]['hard'].append(hard)


#%%
df_delta = pd.DataFrame.from_dict(dict_delta)

#%%
plt.figure(figsize=(10, 6))

# Dégradé de violettes
pastel_colors = ["#D5E8D4",  # Vert forêt clair
                   "#66BB6A",  # Vert forêt moyen  
                   "#3D8B3D"]

sns.boxplot(data=df_delta, palette=pastel_colors)

# Stripplot avec la même palette pour éviter les doublons
sns.stripplot(data=df_delta, palette=pastel_colors, alpha=0.7, size=6, edgecolor='gray', linewidth=0.5, dodge=True)

plt.title('Boxplot with Individual Data Points')
plt.ylabel('Delta Values')
plt.show()

for col in df_delta.columns:
    print('moyenne:', df_delta[col].mean(), '+/-', df_delta[col].std())
    
#%% LMM
input_data = df.query("Condition in ['Hard', 'Easy']")
input_data = input_data.query("Chroma in ['hbo']")
# input_data = input_data.query("ROI in ['Center','Right_Hemisphere','Left_Hemisphere']")
input_data["ROI"] = pd.Categorical(
    input_data["ROI"],
    categories=["Center", "Left_Hemisphere", "Right_Hemisphere"]
)
input_data["ROI"] = input_data["ROI"].cat.reorder_categories(
    ["Left_Hemisphere", "Center", "Right_Hemisphere"],
    ordered=False
)
roi_model = smf.mixedlm("Value ~ Condition*ROI", input_data, groups=input_data["ID"]).fit()

roi_model.summary()