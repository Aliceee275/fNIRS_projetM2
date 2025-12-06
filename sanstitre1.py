# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 23:35:19 2025

@author: gauta

In this script, we analyze the Nirs data of 4 participants solving hard 
and easy math questions. 

Our goal is to determine which ROI is the most sensitive to difficulty.

This script is based on our classes, the plot_16_waveform_group.ipynb,
and follows the instruction of our project found on moodle.

The mne and mne_nirs, as bell as graph related libraries are required.
A config script containing the paths is also necessary, with the 
necessary changes to work on your computer.

This script follows the PEP8 rules
"""

#%% Import librairies
#standard imports
import glob
import os

from collections import defaultdict
from copy import deepcopy
from itertools import compress

# plotting and statistic analysis libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

#non standard imports
import mne
from mne import Epochs, events_from_annotations, set_log_level
from mne.preprocessing.nirs import (
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
    temporal_derivative_distribution_repair,
)
from mne.viz import plot_compare_evokeds

from mne_nirs.channels import get_long_channels, picks_pair_to_idx
from mne_nirs.signal_enhancement import enhance_negative_correlation

#local imports
import config as cfg

# Set general parameters
# Don't show info, as it is repetitive for many subjects
set_log_level("WARNING")  

#%%
"""

We follow the step as seen in class, and add the steps from the 
plot_16_waveform_group.ipynb and the project on Moodle:
    
1- Load the data (Nirx Format)
1bis - Rename trigger 1 and 2 in Hard and Easy

2- Convert intensity to OD
2bis - Downsample to 2Hz

3- Artifact correction (TDDR)

4- Convert OD to concentration

5- Filtering with pass band ([0.002 - 0.3] Hz)

6- Create epochs [-5; +30] ms around the events

7- Trial averaging (not in the individual_analysis function)

"""

#%%define the analysis function


def individual_analysis(path):
    # 1- Load the data (Nirx Format)
    raw_intensity= mne.io.read_raw_nirx(
        path,
        verbose=False
        )
    raw_intensity = get_long_channels(
        raw_intensity,
        min_dist=0.01
        )
    
    # 1bis - Rename trigger 1 and 2 in Hard and Easy
    raw_intensity.annotations.rename({'1.0': 'Hard', '2.0':'Easy'})
    #visualize (optional, for the presentation)
    #raw_intensity.plot(duration=30)
    
    # 2- Convert intensity to OD and determine bad channels
    raw_od = optical_density(raw_intensity)
    sci = scalp_coupling_index(
        raw_od,
        h_freq=1.35,
        h_trans_bandwidth=0.1
        )
    raw_od.info["bads"] = list(compress(raw_od.ch_names, sci < 0.5))
    raw_od.interpolate_bads()
    #visualize
    #raw_od.plot(duration=30)
    
    # 2bis- Down sample to 2 Hz
    raw_od.resample(2)
    #visualize
    #raw_od.plot(duration=30)
    
    #3 - Artifact correction : apply movement correction (TDDR)
    raw_od = temporal_derivative_distribution_repair(raw_od)
    
    #visualize
    #raw_od.plot(duration=30)

    # 4- Convert OD to concentration (haemoglobin) 
    raw_haemo = beer_lambert_law(
        raw_od,
        ppf=0.1
        )
    #visualize
    #raw_haemo.plot(duration=30)
    # 5- Filtering with pass band ([0.002 - 0.3] Hz)
    raw_haemo = raw_haemo.filter(
        0.02,
        0.3,
        h_trans_bandwidth=0.1,
        l_trans_bandwidth=0.01,
        verbose=False
    )
    #visualize
    #raw_haemo.plot(duration=30)

    # Apply further data cleaning techniques (mne tutorial) and extract epochs
    raw_haemo = enhance_negative_correlation(raw_haemo)
    # Extract events
    events, event_dict = events_from_annotations(
        raw_haemo
        )
    #Create epochs
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
#define the paths of the Nirx files
path_root=cfg.rootpath
path_data=os.path.join(path_root, "NIRS_DATA")
path_raw=os.path.join(path_data, "NIRS_Raw_Data")

files= glob.glob(os.path.join(path_raw, "*"))

#Initiate a dictionnary to store the evokeds for each condition
all_evokeds = defaultdict(list)

#run the analysis for each participant
for i,file in enumerate(files):
    path= file
    subject= file.split("_")[-1]
    datatype= "nirs"
    # extract the epochs and raw_haemo for each participant
    raw_haemo, epochs = individual_analysis(path)

    # Save individual-evoked participant data along with others in all_evokeds
    for cidx, condition in enumerate(epochs.event_id):
        all_evokeds[condition].append(epochs[condition].average())
        
print(all_evokeds)

#%% 
"""

Then we move on to the analysis part:

First, we visualize the temporal discourse of hbO and hHb,
with a mean of all channels for both conditions

Then, we visualize the same thing but for out 3 ROIs: left, center and right

Finally we compute the max amplitude for each participant for bhO and hHb 

The goal is to determine which ROI is the most affected 
by the effort difference between Hard and Easy

Therefore, for each ROI we compute the mean difference between conditions 
for all participants and visualize this

"""

#%%Visualisation of Hbo and Hbr for each channels for easy and hard
fig, axes = plt.subplots(nrows=1, ncols=len(all_evokeds), figsize=(15, 8))

#To have the same scale everywhere, we set the limits of the y axis
lims = dict(hbo=[-10, 20], hbr=[-10, 20])

#For both hbo and hbr(=hHb)
for pick, color in zip(["hbo", "hbr"], ["r", "b"]):
    #For each condition of evoked (all the epochs in easy and all the epochs in hard)
    for idx, evoked in enumerate(all_evokeds):
        #We use the mne function to visualize both evokeds (hbo and hbr)
        plot_compare_evokeds(
            {evoked: all_evokeds[evoked]},
            combine="mean",
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
    [6,5], [6,6], #source S6
    [7,5], [7,7], #source S7
    [8,6], [8,7], #source S8
]

# Then generate the correct indices for each pair and store in dictionary rois
rois = dict(
    Left_Hemisphere=picks_pair_to_idx(raw_haemo, left),
    Center=picks_pair_to_idx(raw_haemo, center),
    Right_Hemisphere=picks_pair_to_idx(raw_haemo, right),
)

print(rois)

#Visualisation
fig, axes = plt.subplots(nrows=len(rois), ncols=len(all_evokeds), figsize=(12, 10))
lims = dict(hbo=[-10, 20], hbr=[-10, 20])

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
#First we create a df and compute the maximal amplitudes
df = pd.DataFrame(columns=["ID", "ROI", "Chroma", "Condition", "Amplitude"])

for idx, evoked in enumerate(all_evokeds):
    subj_id = 0
    for subj_data in all_evokeds[evoked]:
        subj_id += 1
        for roi in rois:
            for chroma in ["hbo", "hbr"]:
                data = deepcopy(subj_data).pick(picks=rois[roi]).pick(chroma)
                #We get the maximal amplitude value around the supposed peak (at 6 sec post stimulus) 
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
                #We add this_df to the pre existing df 
                df = pd.concat([df, this_df], ignore_index=True)

df.reset_index(inplace=True, drop=True)
df["Value"] = pd.to_numeric(df["Value"])

    
#%% Visualize max amplitudes for each participant for Hbo and Hbr with a lineplot 
chromas= ['hbo', 'hbr']

#First we set the limits of the y axis to have the same scale for each figure
y_min = df.loc[df['Chroma']=='hbo', 'Value'].min()
y_max = df.loc[df['Chroma']=='hbo', 'Value'].max()+1

#To have all figures in the same window: subplot for each ROI and chroma
fig, axes = plt.subplots(nrows=len(chromas), ncols=len(rois), figsize=(14, 10))

for row , chrom in enumerate(chromas):
    for col, roi in enumerate(rois):
        df_roi = df.loc[df['ROI']==roi].query("Chroma == @chrom")
        ax = axes[row, col]
        sns.lineplot(
            x="Condition", 
            y="Value",
            hue="ID",
            data=df_roi,
            palette="muted",
            marker='o',
            ax=ax, 
            legend=True,
        )
        ax.set_ylim(y_min, y_max) 
        #put the titles for each column and row
        ax.set_title(f"ROI: {roi}") 
        ax.set_ylabel(chrom)

plt.tight_layout()  
plt.show()

#%%Visualize amplitude difference: hard/easy conditions for each participant

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
        
df_delta = pd.DataFrame.from_dict(dict_delta)

# Visualize
fig, axes = plt.subplots(nrows=len(chromas), ncols=1, figsize=(7, 10))

# we create a color palette for our ROIs to make the boxplot more visual
colors = ["#D5E8D4", "#66BB6A", "#3D8B3D"]
#Then we make a boxplot and a stripplot to see the data of each individual

for row, chrom in enumerate(chromas):
    ax = axes[row]
    sns.boxplot(data=df_delta, palette=colors, ax=ax)
    sns.stripplot(
        data=df_delta,
        palette=colors,
        alpha=0.7,
        size=6,
        edgecolor='gray',
        linewidth=0.5, 
        dodge=True, 
        ax=ax
        )
    #ax.set_ylim(y_min, y_max) 
    ax.set_ylabel(chrom)
    ax.set_title(f'Amplitude difference for each ROI ({chroma})')


plt.tight_layout()
plt.show()

#print the mean for each ROI
for col in df_delta.columns:
    print('mean +/- std:', df_delta[col].mean(), '+/-', df_delta[col].std())

#%% For statistical analysis, we use a LMM (effects of ROI and conditions)
input_data = df.query("Condition in ['Hard', 'Easy']")
input_data = input_data.query("Chroma in ['hbo']")

#We have to put the ROIs as categorical variables for the LMM
input_data["ROI"] = pd.Categorical(
    input_data["ROI"],
    categories=["Center", "Left_Hemisphere", "Right_Hemisphere"]
)

#Reordering data allows to change the reference for the LMM
input_data["ROI"] = input_data["ROI"].cat.reorder_categories(
    ["Left_Hemisphere", "Center", "Right_Hemisphere"],
    ordered=False
)

#we create our model
roi_model = smf.mixedlm("Value ~ Condition*ROI", input_data, groups=input_data["ID"]).fit()

#we print the summary with the statistical data
roi_model.summary()