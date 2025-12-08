fNIRS project - M2 NCI - BAHAZ LEBBAD Imen, GAUTIER Alice, MANDOLI Marina

This file contains all the necessary files to run the fNIRS analyses for the M2 NCI project.

Folders:
NIRS_DATA/ NIRS_Raw_Data contains the data of our 4 participants, with a folder by participant, named with the date of the recording and the participant number. In each folder are 13 files containing part of the NIRS recording data, that are used in our scripts.

Scripts:
config.py - This script is used to store the paths for each new person that has access to the code. To ensure the main code works on a new computer, it is necessary to enter the path of the present file (fNIRS_projetM2) in a {name}_rootpath variable, and to add it in the if loop to select the right path depending on the computer:

elif 'name' in os.getcwd() :
    rootpath = rootpath_name

Replace 'name' with your computer's username.

fNIRS_project.py - This script contains the main code to run the analysis. It analyzes the data of the four participants and visualizes it with several figures to see which ROI is the most impacted by changes in difficulty for arithmetic calculations.
To run this script, you will neec to install the mne, mne_nirs, pandas and seaborn libraries.