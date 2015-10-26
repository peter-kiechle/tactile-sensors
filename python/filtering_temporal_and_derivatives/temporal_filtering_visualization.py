# -*- coding: utf-8 -*-

# Load configuration file before pyplot
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

# Library path
import os, sys
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)

import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma

from pykalman import KalmanFilter
from pykalman import UnscentedKalmanFilter

import savitzky_golay as SG1 # from http://www.scipy.org/Cookbook/SavitzkyGolay
import signal_smooth as smooth # from http://scipy.org/Cookbook/SignalSmooth



from scipy.ndimage import filters

import framemanager_python

# Force reloading of external library (convenient during active development)
reload(framemanager_python)


def load_data(filename, matrixID):
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(filename);
    averages = frameManager.get_average_matrix_list(matrixID)
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, averages


def load_data_taxel(filename, matrixID, x, y):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
    texel = frameManager.get_texel_list(matrixID, x, y)
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, texel


def load_data_smoothed(filename, matrixID):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
    #frameManager.set_filter_none()
    #frameManager.set_filter_median(1, False)
    frameManager.set_filter_gaussian(1, 0.85)
    #frameManager.set_filter_bilateral(2, 1500, 1500)
    
    #y = frameManager.get_max_matrix_list(matrixID)
    y = frameManager.get_average_matrix_list(matrixID)

    # Value of centroid taxel
    #featureExtractor = framemanager_python.FeatureExtractionWrapper(frameManager)
    #numFrames = frameManager.get_tsframe_count()
    #y = np.empty([numFrames])
    #for frameID in range(0,numFrames):
    #    centroid = np.rint(featureExtractor.compute_centroid(frameID, matrixID)).astype(int)
    #    y[frameID] = frameManager.get_texel(frameID, matrixID, centroid[0], centroid[1])
    
    # Median of nonzero values
    #numFrames = frameManager.get_tsframe_count()
    #y = np.empty([numFrames])
    #for frameID in range(0,numFrames):
    #    tsframe = frameManager.get_tsframe(frameID, matrixID)
    #    tsframe_nonzero = np.ma.masked_where(tsframe == 0, tsframe) # Restrict to nonzero values
    #    median = np.ma.median(tsframe_nonzero)
    #    y[frameID] = np.nan_to_num(median) # NaN to 0
    
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, y

def load_data_tsframes3D(filename, matrixID):
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(filename);
    # Load entire profile as 3D array
    numTSFrames = frameManager.get_tsframe_count();
    tsframe = frameManager.get_tsframe(0, matrixID);
    width = tsframe.shape[1]
    height = tsframe.shape[0]
    tsframes3D = np.empty((height, width, numTSFrames)) # height, width, depth
    for i in range(numTSFrames):
        tsframes3D[:,:,i] = np.copy( frameManager.get_tsframe(i, matrixID) )
    return tsframes3D



def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx



profileName = os.path.abspath("centrical_87_tennisball.dsa3")
matrixID = 3
#x,y = load_data_taxel(profileName, matrixID, 5, 9); #4, 6
x,y = load_data(profileName, matrixID); #4, 6
tsframes3D = load_data_tsframes3D(profileName, matrixID);



duration = 3.0
start_idx = np.argmax(y!=0)-1
start_time = x[start_idx]
#stop_idx = find_nearest_idx(x, x[start_idx]+duration)
stop_idx = y.shape[0] 
stop_idx_plotting = find_nearest_idx(x, x[start_idx]+duration) - start_idx

# Trim data
x = x[start_idx:stop_idx]-start_time
y = y[start_idx:stop_idx]
tsframes3D = tsframes3D[:,:,start_idx:stop_idx]

numTSFrames = x.shape[0]


names = ['running_mean',
         'convolved', 
         'savitzky_golay', 
         'savitzky_golay_1', 
         'savitzky_golay_2', 
         'gaussian',
         'kalman',
         'pykalman-zero-order',
         'pykalman-2nd-order', 
         'pykalman_smoothed-zero-order',
         'pykalman_smoothed-2nd-order',
         'unscented_kalman',
         'per_taxel_kalman',
         'tv',
         'lowess' ]

formats = ['float' for n in names]
dtype = dict(names = names, formats=formats)
y_filtered = np.empty((numTSFrames), dtype=dtype)




#--------------------------------------
# Running mean (using convolution)
#--------------------------------------
N = 5
y_filtered['running_mean'] = np.convolve(np.copy(y), np.ones((N,))/N)[(N-1):]
test = y_filtered['running_mean']



#--------------------------------------
# Using convolution to smooth signal
#--------------------------------------
N = 5
y_filtered['convolved'] = smooth.smooth(np.copy(y), N, window='hanning')[(N-1)/2:-(N-1)/2]



#-------------------
# Gaussian filter
#------------------
y_filtered['gaussian'] = filters.gaussian_filter1d(np.copy(y), 3.0, mode='nearest', cval=0.0, order=0)




#----------------------------------------------------------------
# My Kalman filter implementation without using PyKalman library
# 2nd order (mass-spring-damper)
#-----------------------------------------------------------------
m = 1000.0 # mass
k_s = 0.05 # spring (the larger, the harder)
k_d = 0.5 # damper (the larger, the harder)
d = k_d / k_s
print("d: " + str(d) + ",  Damped: " + str(d > 1))

q = 0.1  # Variance of process noise, i.e. state uncertainty
r = 0.01 # Variance of measurement error

# Posterior state estimate, \hat{x}_{k}
initial_state_mean = np.matrix([[0.0],
                                [0.0]])
#initial_state_mean = np.array([0.0,0.0])                               
                                
n = initial_state_mean.size # states 


# State error covariance P
# The error matrix (or the confidence matrix): P states whether we should
# give more weight to the new measurement or to the model estimate
# sigma_model = 1;
# P = sigma^2*G*G';
initial_state_covariance = np.matrix([[500.0, 100.0],
                                      [100.0, 500.0]])
 
# F (A), state transition matrix
transition_matrix = np.matrix([[0.0, 1.0],
                              [-k_s/m, -k_d/m]]) 

# b (B, G)
transition_offset = np.matrix([[0],
                               [1/m]])

# C (H, M)
observation_matrix =  np.matrix([[1, 0]]) 

# d (D)
observation_offset = 0.0 

# Coupled noise parameters: proper method would be Autocovariance Least-Squares (ALS)
# Q, covariance of process noise (no control input: Q = 0)
transition_covariance = q * np.eye(n)

# R, covariance matrix of measurement error (= sample variance in this case)
observation_covariance = r 

# Rename
F = transition_matrix
x_posterior_estimate = initial_state_mean
P_posterior = initial_state_covariance
Q = transition_covariance
R = observation_covariance
H = observation_matrix
B = transition_offset
I = np.eye(n)

y_kalman = np.empty([y.shape[0]])
for i in xrange(1, y.shape[0]):
    #----------------------------------
    # Prediction (time update)
    #----------------------------------
    x_prior_estimate = F * x_posterior_estimate #+ np.matrix([[y[i]], [0]]) #+ B*y[i] # Predict next state
    P_prior = F * P_posterior * F.T + Q # Predict next error covariance

    # Perform measurement
    z = y[i]  #noisy_measurement[i]
    
    #----------------------------------
    # Correction (measurement update)
    #----------------------------------
    residual = z - (H*x_prior_estimate) 
    S = H * P_prior * H.T + R
    K = (P_prior * H.T) * np.linalg.pinv(S) # Kalman gain
    x_posterior_estimate = x_prior_estimate + ( K * residual )  # Update state estimate
    #P_posterior = P_prior - K*S*K.T; # Update error covariance
    P_posterior = (I - (K*H)) * P_prior # Update error covariance
    
    # Save result
    y_kalman[i] = x_posterior_estimate[0]


y_filtered['kalman'] = y_kalman

'''
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y, color='r', alpha=0.5, label='noisy measurements')
ax.plot(y_kalman, 'b', alpha=0.5, label='filtered')
ax.legend(loc="upper right")

fig.tight_layout()
plotname = "temporal_filtering_kalman"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
plt.close()
'''



'''
#--------------------------------------------------------------------------------------------
# Example PyKalman library
#--------------------------------------------------------------------------------------------
# Learn good values for parameters named in `em_vars` using the EM algorithm
loglikelihoods = np.zeros(5)
for i in range(len(loglikelihoods)):
    kf = kf.em(X=y, n_iter=1)
    loglikelihoods[i] = kf.loglikelihood(y)


import pylab as pl  
# Draw log likelihood of observations as a function of EM iteration number.
# Notice how it is increasing (this is guaranteed by the EM algorithm)
pl.figure()
pl.plot(loglikelihoods)
pl.xlabel('em iteration number')
pl.ylabel('log likelihood')
pl.show()

# Estimate the hidden states using observations up to and including
# time t for t in [0...n_timesteps-1].  This method outputs the mean and
# covariance characterizing the Multivariate Normal distribution for
#   P(x_t | z_{1:t})
filtered_state_estimates = kf.filter(y)[0]

# Estimate the hidden states using all observations.  These estimates
# will be 'smoother' (and are to be preferred) to those produced by
# simply filtering as they are made with later observations in mind.
# Probabilistically, this method produces the mean and covariance
# characterizing,
#    P(x_t | z_{1:n_timesteps})
smoothed_state_estimates = kf.smooth(y)[0]



#states_pred = kf.em(y, n_iter=5)

#states_pred = kf.em(y).smooth(y)[0]
#print('fitted model: {0}'.format(kf))

#y_kalman = kf.filter(y)[0][:,0]

#y_filtered['kalman'] = kf.filter(y)[0].flatten()

'''



#--------------------------------------------------------------------------------------------
# Kalman filter 0-order (PyKalman)
#--------------------------------------------------------------------------------------------
transition_matrix = 1.0 # F
transition_offset = 0.0 # b
observation_matrix = 1.0 # H
observation_offset = 0.0 # d
transition_covariance = 0.05 # Q
observation_covariance = 0.5  # R

initial_state_mean = 0.0
initial_state_covariance = 1.0

# sample from model
kf = KalmanFilter(
    transition_matrix, observation_matrix, transition_covariance,
    observation_covariance, transition_offset, observation_offset,
    initial_state_mean, initial_state_covariance,
)

#kf = kf.em(y, n_iter=5) # Expectation Maximization
y_filtered['pykalman-zero-order'] = kf.filter(y)[0].flatten()
y_filtered['pykalman_smoothed-zero-order'] = kf.smooth(y)[0][:,0]




#--------------------------------------------------------------------------------------------
# Kalman filter 2nd order (mass-spring-damper) (PyKalman)
#--------------------------------------------------------------------------------------------
m = 1000.0 # mass
k_s = 0.05 # spring (the larger, the harder)
k_d = 0.5 # damper (the larger, the harder)
d = k_d / k_s
print("d: " + str(d) + ",  Damped: " + str(d > 1))

q = 0.1  # Variance of process noise, i.e. state uncertainty
r = 0.01 # Variance of measurement error

# Posterior state estimate, \hat{x}_{k}
initial_state_mean = np.array([0.0,0.0])                               
                                
n = initial_state_mean.size # states 

# State error covariance , P
# The error matrix (or the confidence matrix): P states whether we should
# give more weight to the new measurement or to the model estimate
# sigma_model = 1;
# P = sigma^2*G*G';
initial_state_covariance = np.matrix([[500.0, 100.0],
                                      [100.0, 500.0]])

# F (A), state transition matrix
transition_matrix = np.matrix([[0.0, 1.0],
                              [-k_s/m, -k_d/m]]) 

# b (B, G)
transition_offset = np.array([0.0,1/m])

# C (H, M)
observation_matrix =  np.matrix([[1, 0]]) 
#observation_matrix =  np.array([1.0, 0.0]) 

# d (D)
observation_offset = 0.0 

# Coupled noise parameters: proper method would be Autocovariance Least-Squares (ALS)
# Q, covariance of process noise (no control input: Q = 0)
transition_covariance = q * np.matrix([[1.0, 0.5],
                                       [0.5, 1.0]])

# R, covariance matrix of measurement error (= sample variance in this case)
observation_covariance = r 

# sample from model
kf = KalmanFilter(transition_matrix,
                  observation_matrix,
                  transition_covariance,
                  observation_covariance, 
                  transition_offset, 
                  observation_offset,
                  initial_state_mean, 
                  initial_state_covariance,
                  em_vars=['transition_matrices', 'observation_matrices',
                           'transition_covariance', 'observation_covariance',
                           'observation_offsets', 'initial_state_mean',
                           'initial_state_covariance'
                           ]
)

#kf = kf.em(y, n_iter=5) # Expectation Maximization
y_filtered['pykalman-2nd-order'] = kf.filter(y)[0][:,0]
y_filtered['pykalman_smoothed-2nd-order'] = kf.smooth(y)[0][:,0]






'''
#----------------------------
# Unscented Kalman filter
#----------------------------
def transition_function(state, noise):
    return state + noise

def observation_function(state, noise):
    return state + noise

transition_covariance = 0.05
observation_covariance = 50.0
initial_state_mean = 0.0
initial_state_covariance = 50.0

ukf = UnscentedKalmanFilter(
    transition_function, observation_function,
    transition_covariance, observation_covariance,
    initial_state_mean, initial_state_covariance,
)

#y_filtered['unscented_kalman'] = ukf.filter(y)[0].flatten()

y_kalman= ukf.filter(y)[0].flatten()
'''




'''
#----------------------------
# Per taxel Kalman filter
#----------------------------
width = tsframes3D.shape[1]
height = tsframes3D.shape[0]
tsframes3D_kalman = np.empty((height, width, numTSFrames)) # height, width, depth
for i in range(width):
    for j in range(height):
        # Kalman filter
        transition_matrix = 1.0 # F
        transition_offset = 0.0 # b
        observation_matrix = 1.0 # H
        observation_offset = 0.0 # d
        transition_covariance = 0.05 # Q
        observation_covariance = 0.5  # R

        initial_state_mean = 0.0
        initial_state_covariance = 1.0

        # sample from model
        kf = KalmanFilter(
                transition_matrix, observation_matrix, transition_covariance,
                observation_covariance, transition_offset, observation_offset,
                initial_state_mean, initial_state_covariance,
            )

        #kf = kf.em(y, n_iter=5)
        taxels = tsframes3D[j,i,:]
        tsframes3D_kalman[j,i,:] = np.round( kf.filter(taxels)[0].flatten() )

# Mean of per taxel Kalman filtered matrix
y_filtered['per_taxel_kalman'] = np.empty(numTSFrames)
for i in range(numTSFrames):
    masked = ma.masked_greater(tsframes3D_kalman[:,:,i], 0)
    y_filtered['per_taxel_kalman'][i] = np.mean(tsframes3D_kalman[:,:,i][masked.mask] )
'''




#----------------------------
# 3D Total variation
#----------------------------
from skimage.restoration import denoise_tv_chambolle

y_max = np.max(y)
y_normalized = np.reshape(y, (-1, 1)) / y_max
y_tv = denoise_tv_chambolle(y_normalized, weight=0.8, multichannel=False)[:,0]
y_filtered['tv'] = y_tv * y_max



#----------------------------
# LOWESS (locally weighted scatter plot smooth)
#----------------------------
import statsmodels.api as sm
y_filtered['lowess'] = sm.nonparametric.lowess(np.copy(y), np.copy(x), frac=0.05)[:,1]



#----------------------------
# Savitzky Golay filter
#----------------------------
y_filtered['savitzky_golay'] = SG1.savitzky_golay(np.copy(y), window_size=5, order=2, deriv=0)



#----------------------------
# Derivatives
#----------------------------
y_filtered['savitzky_golay_1']  = SG1.savitzky_golay(y_filtered['gaussian'], window_size=5, order=2, deriv=1)
y_filtered['savitzky_golay_2'] = SG1.savitzky_golay(y_filtered['gaussian'], window_size=5, order=2, deriv=2)





#-------------------------------------------
# Principal Component Analysis (time series)
#-------------------------------------------
# Projected timeseries onto their principal component 
# in order to achieve spatial invariance with respect to the most responsive taxels.
from sklearn.decomposition import PCA

# Extract only the signals of active taxels
threshold = 200
taxel_max = np.max(tsframes3D, axis=2)
n_active = taxel_max[taxel_max > threshold].size
active_idx = np.where(taxel_max > threshold)
width = tsframes3D[:,:,0].shape[1]
height = tsframes3D[:,:,0].shape[0]
tsframes3D_active = np.empty((n_active, numTSFrames)) # height, width, depth

for i in range(n_active):
    tsframes3D_active[i, :] = tsframes3D[active_idx[0][i],active_idx[1][i],:]


tsframes3D_active = tsframes3D_active[:, 1:488]

# Standardize samples (zero mean, one standard deviation)
feature_mean = np.mean(tsframes3D_active, axis=0, keepdims=True)
feature_stddev = np.std(tsframes3D_active, axis=0, keepdims=True)
tsframes3D_active_normalized = (tsframes3D_active - feature_mean) / feature_stddev

pca = PCA(n_components=1)
pca.fit(tsframes3D_active_normalized)
#samples_train_transformed = pca.transform(samples_train)
y_pca_components = pca.components_.T # Attention: Not the same shape as y_filtered





############
# Plotting
###########

x = x[:stop_idx_plotting]
y = y[:stop_idx_plotting]
y_filtered = y_filtered[:stop_idx_plotting]

brewer_red = [0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.21568627, 0.49411765, 0.72156863]
brewer_green = [0.30196078, 0.68627451, 0.29019608]


# Categorical colors
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # Blue
          (1.0, 0.4980392156862745, 0.054901960784313725), # Orange
          (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
          (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # Red
          (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # Purple
          (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
          (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # Greenish/yellow
          (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)] # Aquamarine






text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()



#--------------------------------------------------------------------------------------------
# Temporal filters (misc)
#--------------------------------------------------------------------------------------------
fig = plt.figure(figsize=figure_size)
ax = plt.subplot(111)

ax.plot(x, y, linestyle="", color=[0.2, 0.2, 0.2, 1.0], zorder=10, label='Mean of active taxels',
        marker='o', markersize=4.0, markeredgewidth=0.5, markerfacecolor=[1.0, 1.0, 1.0, 1.0], markeredgecolor=[0, 0, 0, 1.0])

ax.plot(x, y_filtered['running_mean'], '-', linewidth=1.5, color=config.UIBK_orange, alpha=1.0, label='Running Mean (N = 5)')

ax.plot(x, y_filtered['gaussian'], '-', linewidth=1.5, color=[0.0, 0,0, 0.0], alpha=1.0, label=r'Gaussian Filter ($\sigma$ = 3)')

#ax.plot(x, y_filtered['convolved'], '-', linewidth=1.5, color=, alpha=0.5, label='Convolution (N = 5)')

ax.plot(x, y_filtered['tv'], '-', linewidth=1.5, color=brewer_red, alpha=1.0, label='Total Variation')

#ax.plot(x, y_filtered['savitzky_golay'], '-', linewidth=1.5, color=brewer_green, alpha=1.0, label='Savitzky Golay')

ax.plot(x, y_filtered['lowess'], '-', linewidth=1.5, color=brewer_blue, alpha=1.0, label='LOWESS')

#ax.plot(x, y_filtered['per_taxel_kalman'], '-', linewidth=1.5, color=brewer_red, alpha=1.0, label='Per taxel Kalman Filter (zero order)')

#mask = np.isfinite(y_kalman) 
#ax.plot(x[mask], y_kalman[mask], '-', linewidth=0.5, color=config.UIBK_blue, alpha=1.0, label='Kalman Filter')
#ax.plot(x, y_kalman, '-', linewidth=0.5, color=brewer_green, alpha=1.0, label='Kalman Filter')

#ax.plot(x, y_filtered['kalman'], '-', linewidth=1.5, color=config.UIBK_blue, alpha=1.0, label='Kalman Filter (zero order)')



#ax.set_ylim([0, 850])
ax.set_ylabel("Mean Sensor Value", rotation=90)
ax.set_xlabel("Time [s]")
ax.legend(loc = 'lower right', fancybox=True, shadow=False, framealpha=1.0)


#plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.9)
fig.tight_layout()
#plt.show() 

plotname = "temporal_filtering"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()




#--------------------------------------------------------------------------------------------
# PCA
#--------------------------------------------------------------------------------------------
fig = plt.figure(figsize=figure_size)
ax = plt.subplot(111)

x2 = x[1:]
y2 = y[1:]
y_filtered2 = y_filtered[1:]
y_pca_components = y_pca_components[0:stop_idx_plotting-1]

ax.plot(x2, y2/y2.max(), linestyle="", color=[0.2, 0.2, 0.2, 1.0], zorder=10, label='Mean of active taxels',
        marker='o', markersize=4.0, markeredgewidth=0.5, markerfacecolor=[1.0, 1.0, 1.0, 1.0], markeredgecolor=[0, 0, 0, 1.0])
        
ax.plot(x2, y_filtered2['gaussian']/y_filtered['gaussian'].max(), '-', linewidth=1.5, color=[0.0, 0,0, 0.0], alpha=1.0, label=r'Gaussian Filter ($\sigma$ = 3)')
ax.plot(x2, y_pca_components / y_pca_components.max(), '-', linewidth=1.5, color=config.UIBK_orange, alpha=1.0, label='PCA')


#ax.set_ylim([0, 850])
ax.set_ylabel("Mean Sensor Value", rotation=90)
ax.set_xlabel("Time [s]")
ax.legend(loc = 'lower right', fancybox=True, shadow=False, framealpha=1.0)
ax.set_ylim([0, 2100])

#plt.subplots_adjust(top=0.8, bottom=0.1, left=0.1, right=0.9)
fig.tight_layout()
#plt.show() 

plotname = "temporal_filtering_PCA"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()




#--------------------------------------------------------------------------------------------
# Kalman filter
#--------------------------------------------------------------------------------------------
fig = plt.figure()
ax = plt.subplot(111)

ax.plot(y, linestyle="", color=[0.2, 0.2, 0.2, 1.0], zorder=10, label='Mean of active taxels',
        marker='o', markersize=4.0, markeredgewidth=0.5, markerfacecolor=[1.0, 1.0, 1.0, 1.0], markeredgecolor=[0, 0, 0, 1.0])

#ax.plot(y_filtered['kalman'], ls='-', lw=1.5, color='g', alpha=0.95, label='Kalman Filter')

ax.plot(y_filtered['pykalman-zero-order'], ls='-', lw=1.5, color=brewer_red, alpha=1.0, label='Kalman (zero-order)')
ax.plot(y_filtered['pykalman-2nd-order'], ls='-', lw=1.5, color=brewer_blue, alpha=1.0, label='Kalman (2nd-order)')

ax.plot(y_filtered['pykalman_smoothed-zero-order'], ls='-', lw=1.5, color=brewer_red, alpha=0.5, label='Kalman smoothed (zero-order)')
ax.plot(y_filtered['pykalman_smoothed-2nd-order'], ls='-', lw=1.5, color=brewer_blue, alpha=0.5, label='Kalman smoothed (2nd-order)')

ax.legend(loc="lower right")
ax.set_ylim([0, 2100])

fig.tight_layout()
plotname = "temporal_filtering_pykalman"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
plt.close()





#--------------------------------------------------------------------------------------------
# Gaussian filtering + Savitzky-Golay
#--------------------------------------------------------------------------------------------

figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]
fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False, squeeze=True, figsize=figure_size)

# Signal
ax = axes[0]
ax.plot(x, y, color=[0.2, 0.2, 0.2, 1.0], zorder=10, linewidth=0.0, label='Data',
        marker='o', markersize=3.0, markeredgewidth=0.5, markerfacecolor=[1.0, 1.0, 1.0, 1.0], markeredgecolor=[0, 0, 0, 1.0])


#ax.plot(x, y_filtered['savitzky_golay'], '-', linewidth=1.5, color=[0.1, 0.1, 0.1], alpha=1.0)
ax.plot(x, y_filtered['gaussian'], '-', linewidth=1.5, color=[0.1, 0.1, 0.1], alpha=1.0)
bbox_props = dict(boxstyle="round", fc="w", ec=[0.0, 0.0, 0.0], alpha=1.0)
ax.text(0.025, 0.89, r"Gaussian smoothed signal ($\sigma$ = 3)", 
        horizontalalignment='left',
        verticalalignment='center',
        fontsize='medium',
        transform=ax.transAxes, 
        bbox=bbox_props)
ax.set_ylim([0, 2100])

ax.set_ylabel("Mean Sensor Value", rotation=90)


# First Derivative
ax = axes[1]
ax.plot(x, y_filtered['savitzky_golay_1'], '-', linewidth=1.5, color=config.UIBK_blue, alpha=1.0, label="First derivative of smoothed signal")
ax.text(0.975, 0.89, r"First derivative of the smoothed signal", 
        horizontalalignment='right',
        verticalalignment='center',
        fontsize='medium',
        transform=ax.transAxes, 
        bbox=bbox_props)


# Second Derivative
ax = axes[2]
ax.plot(x, y_filtered['savitzky_golay_2'], '-', linewidth=1.5, color=config.UIBK_orange, alpha=1.0, label="Second derivative of smoothed signal")
ax.set_xlabel("Time [s]")

ax.text(0.975, 0.89, r"Second derivative of the smoothed signal", 
        horizontalalignment='right',
        verticalalignment='center',
        fontsize='medium',
        transform=ax.transAxes, 
        bbox=bbox_props)

fig.tight_layout()
#plt.show() 

plotname = "temporal_filtering_derivatives"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
fig.close()
