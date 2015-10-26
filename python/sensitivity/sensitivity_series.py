# -*- coding: utf-8 -*-
import os, sys
print("CWD: " + os.getcwd() )

# Load configuration file before pyplot
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

# Library path
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python
#reload(framemanager_python) # Force reloading of external library (convenient during active development)


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


#--------------------
# Auxilary functions
#--------------------

def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

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
    
    
#--------------------
# Fitting functions
#--------------------

# Aperiodic + drift
def funcPT2(t, K, T, d, m):
    T1 = T / (d + np.sqrt(d*d - 1.0)) # Note: d*d instead of d^2 due to Levenberg-Marquardt
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return K - (K / (T1-T2)) * ( (T1 * np.exp(-t/T1)) - (T2 * np.exp(-t/T2))  )  + m*t

# Aperiodic limit case (d=1)
def funcPT2_limit(t, K, T, m):
   return K * (1.0 - (1.0 + t/T) * (np.exp(-t/T)) )  + m*t
   
def first_derivative(t, K, T, d, m):
    T1 = T / (d + np.sqrt(d*d - 1.0)) # Note: d*d instead of d^2 due to Levenberg-Marquardt
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return -K/(T1-T2) * ( -np.exp(-t/T1) + np.exp(-t/T2) ) + m

def second_derivative(t, K, T, d, m):
    T1 = T / (d + np.sqrt(d*d - 1.0)) # Note: d*d instead of d^2 due to Levenberg-Marquardt
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return -K/(T1-T2) * ( np.exp(-t/T1)/T1 - np.exp(-t/T2)/T2 )

def inflection_point(T, d):
    T1 = T / (d + np.sqrt(d*d - 1.0)) # Note: d*d instead of d^2 due to Levenberg-Marquardt
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return (-1/(T1-T2)) * np.log(T2/T1) * T1 * T2



########################
# Load Pressure Profile
########################

#profileName = os.path.abspath("../../pressure_profiles/sensitivity_test_0.5-0.25-0.5-0.75-1.0-0.5.dsa3")
profileName = os.path.abspath("messreihe_0.1-1.0.dsa")

frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
frameManager.set_filter_gaussian(1, 0.85)

numFrames = frameManager.get_tsframe_count();
starttime = frameManager.get_tsframe_timestamp(0)
stoptime = frameManager.get_tsframe_timestamp(numFrames)

# Time stamps
timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds


matrixID = 1
x = timestamps
#y = frameManager.get_texel_list(matrixID, 2, 8)
y = frameManager.get_average_matrix_list(matrixID)


# Trim data
#duration = 4.0
start_idx = max(np.argmax(y!=0)-1, 0) # Index of first non-zero value (or 0)
start_time = x[start_idx]
#stop_idx = find_nearest_idx(x, x[start_idx]+duration) 
stop_idx = x.shape[0]

x = x[start_idx:stop_idx]-start_time
y = y[start_idx:stop_idx]
x_diff = np.diff(x)


# Find partitions of different sensitivies
slice_idx = np.nonzero(x_diff > 1.0)[0]
begin_idx = np.insert(slice_idx+1, 0, 0)
end_idx = np.append(slice_idx+1, y.shape[0])



#---------------------------------
# Simple step detection algorithm
#---------------------------------
# Find all non-zero sequences
# Throw small sequencs away. Actual grasps are remaining
# For more elaborated methods: http://en.wikipedia.org/wiki/Step_detection

thresh_sequence = 100 # Minimum length of a sequence to be considered a "grasp"
tail_length = 100 # The last frame of the grasp will be within this tail sequence
grasp_idx = []
for start, stop in contiguous_regions(y != 0):
    if (stop-start) > thresh_sequence:
        slice_tail = y[stop-tail_length:stop]
        end_position = (stop-tail_length) + find_nearest_idx(slice_tail, np.max(slice_tail))   
        grasp_idx.append([start-1, end_position])

nGrasps = len(grasp_idx)
Ks = np.empty([nGrasps])
Ts = np.empty([nGrasps])
ds = np.empty([nGrasps])
ms = np.empty([nGrasps])

for i in range(nGrasps):

   # Trim data
   #duration = 4.0
   #start_idx = max(np.argmax(y!=0)-1, 0) # Index of first non-zero value (or 0)
   #start_time = x[start_idx]
   #stop_idx = find_nearest_idx(x, x[start_idx]+duration) 


   x_grasp = x[grasp_idx[i][0]:grasp_idx[i][1]]
   x_grasp -= x_grasp[0] # Begin at t=0
   y_grasp = y[grasp_idx[i][0]:grasp_idx[i][1]]
   
   # Scale data for better fitting performance
   max_y = np.max(y_grasp)
   scale_factor_x = max_y / x_grasp[-1]
   x_fitting = x_grasp*scale_factor_x # scaled version


   ###########
   # Fit PT-2
   ###########
   # Initial values
   K = max_y - 0.1*max_y
   T = 0.01 * scale_factor_x
   d = 1.05
   m = 10.0 / scale_factor_x # Todo: Take this parameter from table obtained by linear regression
   p0 = [K, T, d, m]
 
   # Levenberg-Marquardt
   opt_parms, parm_cov = optimize.curve_fit(funcPT2, x_fitting, y_grasp, p0, maxfev=10000)

   K = opt_parms[0]
   T = opt_parms[1] / scale_factor_x
   d = opt_parms[2]
   m = opt_parms[3] * scale_factor_x
   print("Grasp: {},  K: {}, T: {}, d:{}, m: {}".format(i, K, T, d, m))

   # Store values of single experiment
   Ks[i] = K
   Ts[i] = T
   ds[i] = d
   ms[i] = m



Ks2 = np.empty(nGrasps)
for i in range(nGrasps):
    Ks2[i] = (i+1)*0.1*np.max(Ks)


#############
# Plotting
############
'''
# Color...
import matplotlib as mpl
import matplotlib.cm as cm

# Custom colormap UIBK Orange
cdict = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),

        'green': ((0.0, 1.0, 1.0),
                  (1.0, 0.5, 0.5)),

        'blue': ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0))}
                
plt.register_cmap(name='UIBK_ORANGES', data=cdict)

norm = mpl.colors.Normalize(vmin=0, vmax=10)
cmap = plt.get_cmap('UIBK_ORANGES')

m = cm.ScalarMappable(norm=norm, cmap=cmap)

for i in range(10, -1, -1):
   print m.to_rgba(i)
'''
#colorMap = plt.get_cmap('Spectral_r')
colorMap = plt.get_cmap('YlOrRd_r')
colorNorm = plt.Normalize(vmin=0, vmax=nGrasps)
scalarMap = plt.cm.ScalarMappable(cmap=colorMap, norm=colorNorm)
colors = scalarMap.to_rgba(np.arange(0, nGrasps+1))
#colors_alpha = np.copy(colors)
#colors_alpha[:][3]=0.75

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 1.0 #0.48
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()


fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.add_subplot(111)

for i in range(nGrasps):
   xs = np.linspace(0.0, 4, 100)
   ys = funcPT2(xs, Ks[i], Ts[i], ds[i], ms[i])
   ax.plot(xs, ys, ls="-", color=colors[i], label=str(0.1+i*0.1), zorder=0)

   x_grasp = x[grasp_idx[i][0]:grasp_idx[i][1]]
   x_grasp -= x_grasp[0] # Begin at t=0
   y_grasp = y[grasp_idx[i][0]:grasp_idx[i][1]]
   stop = find_nearest_idx(x_grasp, 4)
   #ax.plot(x_grasp[0:stop], y_grasp[0:stop], ls="", color=colors[i], alpha=1.0, 
   #   marker='o', markersize=1.5, markeredgewidth=0.2, markerfacecolor=colors[i], markeredgecolor=(0, 0, 0))

   ax.plot(x_grasp[0:stop], y_grasp[0:stop], ls="", color=colors[i], alpha=1.0, 
      marker='o', markersize=3.0, markeredgewidth=0.3, markerfacecolor=colors[i], markeredgecolor=[0.0, 0.0, 0.0] )



ax.set_ylim([0, 650])

ax.legend(loc='upper left', fancybox=True, shadow=False, framealpha=1.0, #prop={'size':5}
          ncol=2, columnspacing=2.0, labelspacing=0.5, handlelength=1.5, handletextpad=0.5, borderpad=0.5)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Mean Sensor Value", rotation=90)

fig.tight_layout()

#plt.show()

plotname = "sensitivities"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()

