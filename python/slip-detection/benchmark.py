#-------------------
# Benchmark
#-------------------

import time
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)

import numpy as np


# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python
import module_normalized_cross_correlation as NCC

# Force reloading of external library (convenient during active development)
#reload(IM)
reload(NCC)
#reload(framemanager_python)


matrixID = 1
startID = 13 # slip_and_rotation_teapot_handle
stopID = 93 # slip_and_rotation_teapot_handle

profileName = os.path.abspath("slip_and_rotation_teapot_handle.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);


# Load entire profile such that nothing has to be decoded
tsframe = frameManager.get_tsframe(startID, matrixID);
width = tsframe.shape[1]
height = tsframe.shape[0]
tsframes3D = np.empty((height, width, stopID-startID)) # height, width, depth
for i in xrange(startID, stopID):
    tsframes3D[:,:,i-startID] = np.copy( frameManager.get_tsframe(i, matrixID) )


frame_range = stopID-startID
len_sequence = 1000
# between [0 and num_frames[
rand_frame_idx = np.random.randint(frame_range, size=len_sequence)

# Records
slipvectors = np.zeros([len_sequence,2])
slipvectors_ncc_1 = np.zeros([len_sequence,2])
slipvectors_ncc_2 = np.zeros([len_sequence,2])
slipvectors_pc = np.zeros([len_sequence,2])
centroids = np.zeros([len_sequence,2])

elapsed = np.zeros([5,1])


frame0 = tsframes3D[:,:,rand_frame_idx[0]]

# Alcazar
start = time.clock()
import scipy.signal
# Frame dimensions
cols = frame0.shape[1]
rows = frame0.shape[0]    
cols_C = 2*cols-1
rows_C = 2*rows-1
# Indices of corresponding taxel position in C
A = np.tile(np.arange(1.0, cols_C+1), (cols_C, 1)) - (cols_C+1)/2 # Repeat rows and substract zeroing offset
B = np.tile(np.arange(1.0, rows_C+1), (rows_C, 1)).T - (rows_C+1)/2 # Repeat columns and substract zeroing offset
########################################################
## Convolution of reference frame with itself
########################################################
C_stationary = scipy.signal.convolve2d(frame0, frame0, mode='full', boundary='fill', fillvalue=0)
means_columns= np.mean(C_stationary, 0, keepdims=True) # Means of columns (along y-axis)
means_rows = np.mean(C_stationary, 1, keepdims=True) # Means of rows (along x-axis) 
shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication
displacement0 = np.array([shift_x, shift_y])


for frameID in xrange(0, len_sequence):
    frame1 = tsframes3D[:,:,rand_frame_idx[frameID]]
    ########################################################
    ## Convolution of reference frame with comparison frame
    ########################################################
    C_moving = scipy.signal.convolve2d(frame0, frame1, mode='full', boundary='fill', fillvalue=0)
    means_columns = np.mean(C_moving, 0, keepdims=True) # Means of columns (along y-axis)
    means_rows = np.mean(C_moving, 1, keepdims=True) # Means of rows (along x-axis) 
    shift_x = np.mean( (np.dot(A,means_columns.T)) / np.sum(means_columns) ) # np.dot performs matrix multiplication
    shift_y = np.mean( (np.dot(means_rows.T, B)) / np.sum(means_rows) ) # np.dot performs matrix multiplication
    displacement1 = np.array([shift_x, shift_y])
    slipvectors[frameID] = displacement1 - displacement0
    displacement0 = displacement1
    C_stationary = C_moving
elapsed[0] = (time.clock() - start)

# NCC 1
start = time.clock()
for frameID in xrange(0, len_sequence):
    frame1 = tsframes3D[:,:,rand_frame_idx[frameID]]
    slipvectors_ncc_1[frameID] = NCC.normalized_cross_correlation2(frame0, frame1)
    frame0 = frame1
elapsed[1] = (time.clock() - start)

# NCC 2
start = time.clock()
for frameID in xrange(0, len_sequence):
    frame1 = tsframes3D[:,:,rand_frame_idx[frameID]]
    slipvectors_ncc_2[frameID] = NCC.normalized_cross_correlation3(frame0, frame1)
    frame0 = frame1
elapsed[2] = (time.clock() - start)

# PC
start = time.clock()
for frameID in xrange(0, len_sequence):
    frame1 = tsframes3D[:,:,rand_frame_idx[frameID]]
    slipvectors_pc[frameID] = NCC.normalized_cross_correlation4(frame0, frame1)
    frame0 = frame1
elapsed[3] = (time.clock() - start)

# Centroid
start = time.clock()
for frameID in xrange(0, len_sequence):
    frame1 = tsframes3D[:,:,rand_frame_idx[frameID]]
    centroids[frameID] = NCC.normalized_cross_correlation5(frame0, frame1)
    frame0 = frame1
elapsed[4] = (time.clock() - start) 


