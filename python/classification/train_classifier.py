# -*- coding: utf-8 -*-
import os, sys
import numpy as np
from scipy import stats

from sklearn import svm
from sklearn.lda import LDA
from sklearn.externals import joblib

# Library path
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python


# --------------------
# Auxiliary functions
# --------------------
def eval_if_defined(var):
    if var in vars() or var in globals():
        return var
    else:
         return False

def find_nearest(array, value):
    idx = np.argmin(np.abs(array - value))
    return array[idx]

def find_nearest_idx(array, value):
    idx = np.argmin(np.abs(array - value))
    return idx

def listdir_fullpath(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def get_immediate_subdirectories(path):
    return sorted([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))])

def get_profiles(path):
    return sorted([f for f in listdir_fullpath(path) if f.endswith(".dsa") and os.path.isfile(f)])


# Taken from http://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array
# Author: Joe Kington
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx





# ------------------------------------------------------------------
# Prepare relevant pressure profiles and corresponding class labels
# ------------------------------------------------------------------
def list_pressure_profiles(profile_folder):
    profiles = []

    subfolders = get_immediate_subdirectories(profile_folder)
    for folder in subfolders:
        class_folder = os.path.join(profile_folder, folder)
        class_profiles = get_profiles(class_folder)
        profiles.extend(class_profiles)
       
    return profiles


# ------------------------------------------
# Compute and store / load training samples 
# ------------------------------------------
def provide_training_data(training_profiles, frameManager, featureExtractor, recompute_features=False, save_features=True):
    global training_samples_raw
    global training_sample_ids
    global training_labels
    global training_categories
    
    if not eval_if_defined('features_available'): # Skip the rest if features are already available
        # Try to load precomputed features from file
        loading_failed = False
        if not recompute_features:
            try:
                # Load training samples from disk using scikit's joblib (replacement of pickle)
                training_samples_dict = joblib.load("dumped_training_samples.joblib.pkl")
                training_samples_raw = training_samples_dict['training_samples_raw']
                training_sample_ids = training_samples_dict['training_sample_ids']
                training_labels = training_samples_dict['training_labels']
                training_categories = training_samples_dict['training_categories']           
                loading_failed = False
                print("Loading dumped_training_samples.npy!") 
            except IOError:
                print("Loading dumped_training_samples.npy failed!") 
                loading_failed = True

        if recompute_features or loading_failed:
            training_samples_raw, training_sample_ids, training_labels, training_categories = create_samples(training_profiles, frameManager, featureExtractor)
            if save_features:
                # Save training samples using scikit's joblib (replacement of pickle)
                training_samples_dict = {'training_samples_raw': training_samples_raw,
                                         'training_sample_ids' : training_sample_ids, 
                                         'training_labels' : training_labels,
                                         'training_categories' : training_categories}

                joblib.dump(training_samples_dict, "dumped_training_samples.joblib.pkl")
                
                # Dump a human readable version
                readable_str = np.array(["% 3.5f" % n for n in training_samples_raw.reshape(training_samples_raw.size)])
                readable_str = readable_str.reshape(training_samples_raw.shape)
                merged_for_readability = np.hstack([training_sample_ids.reshape(-1, 1), training_labels.reshape(-1, 1), readable_str])
                np.savetxt('dumped_training_samples_human_readable.csv', merged_for_readability, 
                           fmt='%s', delimiter=', ',  comments='# ', newline='\n',
                           header="Human readable raw features. Binary .npy file is used to store actual training data\n"
                                  "Sample ID, Class label, Diameter, Compressibility, "
                                  "StdDev Matrix 1, StdDev Matrix 5, " 
                                  "25 x CM Matrix 1,,,,,,,,,,,,,,,,,,,,,,,,, 25 x CM Matrix 5" )
                                  
                print("Computed training samples saved!")
                  
        global features_available
        features_available = True

    return np.copy(training_samples_raw), training_labels, training_categories # Return a copy since samples are not rescaled yet



# -------------------------------------------------------------------------------------------------------
# Extracts features from recorded profiles, builds feature vectors and combines them to training samples
# -------------------------------------------------------------------------------------------------------
# |-> Diameter of minimal bounding sphere
# |-> Compressibility / rigidity
# |-> Standard deviation
# |-> Chebyshev moments
def create_samples(profiles, frameManager, featureExtractor):
    pmax = 5 # Max moment order
    n_features = 1 + 1 + 2*1 + 2*pmax*pmax
    categories = {} # Classes and number of its members 
    labels = [] # Class membership for each sample grasp
    sample_ids = [] # Individual grasp
    samples = np.empty((0, n_features)) 
  
    for i, profile in enumerate(profiles):
     
        frameManager.load_profile(profile)
        frameManager.set_filter_none()
        #frameManager.set_filter_median(1, True)

        #---------------------------------
        # Simple step detection algorithm
        #---------------------------------
        # Find all non-zero sequences of both tactile sensor matrices
        # Throw small sequencs away. Actual grasps are remaining
        # For more elaborated methods: http://en.wikipedia.org/wiki/Step_detection
        numTSFrames = frameManager.get_tsframe_count();
        max_matrix_1 = frameManager.get_max_matrix_list(1)
        max_matrix_5 = frameManager.get_max_matrix_list(5)
        
        valid_contacts = np.empty([numTSFrames])
        valid_contacts.fill(False)
        for frameID in xrange(0, numTSFrames):    
            if (max_matrix_1[frameID] > 0.0 and max_matrix_5[frameID] > 0.0) :
                valid_contacts[frameID] = True


        thresh_sequence = 20 # Minimum length of a sequence to be considered a "grasp"
        grasps = []
        for start, stop in contiguous_regions(valid_contacts):
            if (stop-start) > thresh_sequence:
                grasps.append([start, stop-1])
                
        num_grasps = len(grasps)
        profile_subfolder = os.path.basename(os.path.dirname(profile))
        class_name = (profile_subfolder.replace("_", " ")) # Class name is defined by the last subfolder's name
        if categories.has_key(class_name):
            categories[class_name] += num_grasps
        else:
            categories[class_name] = num_grasps

        # Compute features for each detected grasp in profile
        for grasp in grasps:

            (grasp_diameter,
             compressibility,
             std_dev_matrix_1, 
             std_dev_matrix_5,
             moments_matrix_1, 
             moments_matrix_5) = compute_features(frameManager, featureExtractor, grasp[0], grasp[1], pmax, max_matrix_1, max_matrix_5)

            # Combine features
            sample = np.concatenate(( [grasp_diameter],
                                      [compressibility],
                                      [std_dev_matrix_1],
                                      [std_dev_matrix_5],
                                      moments_matrix_1,
                                      moments_matrix_5 )).reshape(1, n_features)


            # Add feature vector to sample
            samples = np.vstack([samples, sample])
            
            # Give new sample a name and class mebership
            labels.append(class_name)
            if num_grasps > 1:
                sample_ids.append(profile + "_" + str(grasp[0]) + "-" + str(grasp[1]))
            else:
                sample_ids.append(profile)

    return samples, np.asarray(sample_ids), np.asarray(labels), categories


def compute_features(frameManager, featureExtractor, grasp_begin, grasp_end, pmax, max_matrix_1, max_matrix_5):
    # Values computed in "calibrate_impression_depth.py"
    max_val_matrix_1 = 3554.0
    max_val_matrix_5 = 2493.0
    impression_depth = 1.0 # Just an estimate of the maximal impression in [mm]
    impression_factor_1 = impression_depth / max_val_matrix_1
    impression_factor_5 = impression_depth / max_val_matrix_5 
   
    # Determine more robust frames of interest (begin and end frame of the grasp) 
    # by taking the objects diameter into account

    # head + tail <= thresh_sequence
    head_elem = 10 
    tail_elem = 10
    
    miniballs = np.empty([grasp_end-grasp_begin+1, 4])
    miniballs.fill(None)
    #for i, frameID in enumerate(range(grasp_end-tail_elem+1, grasp_end+1)):
    for i, frameID in enumerate(range(grasp_begin, grasp_end+1)):     
       theta = frameManager.get_corresponding_jointangles(frameID)
       miniballs[i] = featureExtractor.compute_minimal_bounding_sphere_centroid(frameID, theta)

    # Compensate for force dependent sensor matrix impression
    diameter = (2*miniballs[:,3] +
                max_matrix_1[grasp_begin:grasp_end+1]*impression_factor_1 +
                max_matrix_5[grasp_begin:grasp_end+1]*impression_factor_5 )
               
    slice_tail = diameter[-tail_elem:]
    end_position = (grasp_end-tail_elem) + find_nearest_idx(slice_tail, np.median(slice_tail))               


    # Problem: 
    # The object's initial size cannot be measured accurately enough if the grasp applies torque.
    # In that case, the contact surface between object and both sensor matrices is tilted leading to an 
    # overestimation of the real diameter. This asymetry disappears when all forces reach an equilibrium state.
    # In order to get more robust object size features, the profile's centroids of the end position frame
    # is used to recalculate the diameter during each step of the grasp.
    centroid_matrix_1 = featureExtractor.compute_centroid(end_position, 1)
    centroid_matrix_5 = featureExtractor.compute_centroid(end_position, 5)
    points = np.array([ [1.0, centroid_matrix_1[0], centroid_matrix_1[1]],
                        [5.0, centroid_matrix_5[0], centroid_matrix_5[1]]], dtype=np.float64)     
    miniballs_refined = np.empty([grasp_end-grasp_begin+1, 4])
    miniballs_refined.fill(None)
    for i, frameID in enumerate(range(grasp_begin, grasp_end+1)):
        theta = frameManager.get_corresponding_jointangles(frameID)
        miniballs_refined[i] = featureExtractor.compute_minimal_bounding_sphere_points(points, theta)


    # Compensate for force dependent sensor matrix impression
    diameter_refined = (2*miniballs_refined[:,3] +
                        max_matrix_1[grasp_begin:grasp_end+1]*impression_factor_1 +
                        max_matrix_5[grasp_begin:grasp_end+1]*impression_factor_5 )
    
    # Initial position: max diameter of minimal bounding sphere
    slice_head = diameter_refined[0:head_elem]
    initial_position = grasp_begin + np.nanargmax(slice_head)

    # Local indices
    initial_position_grasp = initial_position - grasp_begin
    end_position_grasp = end_position - grasp_begin

    # Compute features
    #grasp_diameter = diameter_refined[initial_position]
    #grasp_diameter = np.median(diameter_refined)
    #grasp_diameter = stats.mode(diameter_refined)[0][0]
    grasp_diameter = stats.mode(diameter)[0][0]
    compressibility = diameter_refined[initial_position_grasp] - diameter_refined[end_position_grasp] # Change of minimal bounding sphere's size during grasp
    std_dev_matrix_1 = featureExtractor.compute_standard_deviation(end_position, 1) # Standard deviation of intensity values (not 2D image moments)
    std_dev_matrix_5 = featureExtractor.compute_standard_deviation(end_position, 5)
    moments_matrix_1 = featureExtractor.compute_chebyshev_moments(end_position, 1, pmax).reshape(-1) # frameID, matrixID, pmax
    moments_matrix_5 = featureExtractor.compute_chebyshev_moments(end_position, 5, pmax).reshape(-1)
        
    return grasp_diameter, compressibility, std_dev_matrix_1, std_dev_matrix_5, moments_matrix_1, moments_matrix_5






#--------------------------------------------
# Main
#--------------------------------------------
training_profile_folder = "training_grasp_profiles_thesis"

frameManager = framemanager_python.FrameManagerWrapper()
featureExtractor = framemanager_python.FeatureExtractionWrapper(frameManager)

# Prepare relevant pressure profiles and corresponding class labels
training_profiles = list_pressure_profiles(training_profile_folder)

# Compute and store / load training samples 
(training_samples, 
training_labels, 
training_categories) = provide_training_data(training_profiles, 
                                             frameManager, 
                                             featureExtractor,
                                             recompute_features=False,
                                             save_features=True)


# ----------------------
# Feature scaling
# ----------------------

# Rescale samples (Range [0,1])
#feature_max = np.max(training_samples, axis=0, keepdims=True)
#feature_min = np.min(training_samples, axis=0, keepdims=True)
#training_samples = (training_samples-feature_min) / (feature_max-feature_min)
#testing_samples = (testing_samples-feature_min) / (feature_max-feature_min)

# Normalize samples (zero mean)
#feature_mean = np.mean(training_samples, axis=0, keepdims=True)
#feature_max = np.max(training_samples, axis=0, keepdims=True)
#feature_min = np.min(training_samples, axis=0, keepdims=True)
#training_samples = (training_samples-feature_mean) / (feature_max-feature_min)

# Standardize samples (zero mean, one standard deviation)
feature_mean = np.mean(training_samples, axis=0, keepdims=True)
feature_stddev = np.std(training_samples, axis=0, keepdims=True)
training_samples = (training_samples - feature_mean) / feature_stddev

#--------------------------------------------------------
# Transform features using Linear Discriminant Analysis
#--------------------------------------------------------
lda = LDA(n_components=14)
lda.fit(training_samples, training_labels)    
training_samples = lda.transform(training_samples)


# --------------------------------------------------------
# Multi-class SVM classification: one-against-one
# --------------------------------------------------------
# There are {n_classes * (n_classes - 1)} classifiers in the one-vs-one scheme
# Order of 0 to n classes:
# 0 vs 1, 0 vs 2, ... 0 vs n, 1 vs 2, ..., 1 vs n, ... n-1 vs n

classifier = svm.SVC( C=100, cache_size=200, class_weight=None, coef0=0.0, degree=3,
                      gamma=0.125, kernel='linear', max_iter=-1, probability=True, random_state=None,
                      shrinking=True, tol=0.001, verbose=False)

# Fit model
classifier.fit(training_samples, training_labels)

# Store fitted classifier and preprocessing steps to disk for later predictions
classifier_dict = {'classifier': classifier, 'means' : feature_mean, 'stddevs' : feature_stddev, 'LDA' : lda}
joblib.dump(classifier_dict, "dumped_classifier.joblib.pkl")
print("Classifier dumped!")


