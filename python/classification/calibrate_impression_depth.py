# -*- coding: utf-8 -*-

# Find maximum sensor values of all recorded profiles

# Load configuration file before pyplot
import os
import numpy as np

import framemanager_python

def listdir_fullpath(path):
    return [os.path.join(path, f) for f in os.listdir(path)]
    
def get_immediate_subdirectories(path):
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

def get_profiles(path):
    return sorted([f for f in listdir_fullpath(path) if f.endswith(".dsa") and os.path.isfile(f)])

def list_pressure_profiles(profile_folder):
    profiles = []

    subfolders = get_immediate_subdirectories(profile_folder)
    for folder in subfolders:
        class_folder = os.path.join(profile_folder, folder)
        class_profiles = get_profiles(class_folder)
        profiles.extend(class_profiles)
       
    return profiles
    
  
training_profile_folder = "training_grasp_profiles_thesis"

frameManager = framemanager_python.FrameManagerWrapper()
featureExtractor = framemanager_python.FeatureExtractionWrapper(frameManager)
frameManager.set_filter_none()

# Prepare relevant pressure profiles and corresponding class labels
training_profiles = list_pressure_profiles(training_profile_folder)

# Iterate through profiles and find maximum sensor value for each matrix
max_matrix_1 = 0
max_matrix_5 = 0
for i, profile in enumerate(training_profiles):
    frameManager.load_profile(profile);
    max_matrix_1 = max(max_matrix_1, np.max(frameManager.get_max_matrix_list(1)))
    max_matrix_5 = max(max_matrix_5, np.max(frameManager.get_max_matrix_list(5)))

print("\n\nMax value matrix 1: {}\nMax value matrix 5: {}".format(max_matrix_1, max_matrix_5))
#Max value matrix 1: 3554.0
#Max value matrix 5: 2493.0
