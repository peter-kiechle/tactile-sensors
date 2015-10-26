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

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize

import framemanager_python
# Force reloading of external library (convenient during active development)
reload(framemanager_python)


def load_data(filename, matrixID):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
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

def load_smoothed(filename, matrixID):
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

def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx



# Proportional Time + Integral element
def funcPT1(t, K, T, m):
   if(m < 0): # ensure m > 0 by assessing a large penalty
       return 1e10
   return K * (1.0 - np.exp(-(t)/T)) + m*t


# Aperiodic limit case
#def funcPT2(t, K, T, m):
#   return K * (1.0 - (1.0 + t/T) * (np.exp(-t/T)) )  + m*t

# Aperiodic + drift
def funcPT2(t, K, T, d, m):
    T1 = T / (d + np.sqrt(d*d - 1.0))
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return K - (K / (T1-T2)) * ( (T1 * np.exp(-t/T1)) - (T2 * np.exp(-t/T2))  )  + m*t



# Load Data
matrixID = 1
x,y = load_data("hockey_ball_pt2.dsa", matrixID) # Good fit
#x,y = load_data("foam_ball_pt2.dsa", matrixID) # Deformation
#x,y = load_data("A1.dsa", matrixID) # Flickering taxels
#x,y = load_data_taxel("golf_ball_001667-001724.dsa", matrixID, 2, 8) # Saturation


# Trim data
duration = 2.0
start_idx = max(np.argmax(y!=0)-1, 0) # Index of first non-zero value (or 0)
start_time = x[start_idx]
stop_idx = find_nearest_idx(x, x[start_idx]+duration) 
#stop_idx = y.shape[0]

# TODO:
# Autodetect duration and/or stop_idx

x = x[start_idx:stop_idx]-start_time
y = y[start_idx:stop_idx]
N = x.shape[0]

# Scale data
max_y = np.max(y)
scale_factor_x = max_y / x[-1]
x_fitting = x*scale_factor_x # scaled version for fitting performance
#x_fitting = np.arange(0, stop_idx-start_idx)*max_y # scaled version for performance reasons


'''
#######################################
# Linear regression (of linear part)
#######################################
from scipy import stats
start_idx_linreg = find_nearest_idx(x, 5.0) # 5 seconds onwards
stop_idx_linreg = y.shape[0] 
x_linreg = x[start_idx_linreg:stop_idx_linreg]
y_linreg = y[start_idx_linreg:stop_idx_linreg]
slope, intercept, r_value, p_value, std_err = stats.linregress(x_linreg,y_linreg)
'''

'''
############
# Fit PT-1
############
K = max_y - 0.1*max_y
T = 0.01 * scale_factor_x
m = 20.0 / scale_factor_x # Take this parameter from table obtained by linear regression
p0 = [K, T, m]

# Levenberg-Marquardt
opt_parms, parm_cov = optimize.curve_fit(funcPT1, x_fitting, y, p0, maxfev=1000)

K = opt_parms[0] # Amplification factor (gain)
T = opt_parms[1] / scale_factor_x # time constant 
m = opt_parms[2] * scale_factor_x
print("K: {}, T: {}, m: {}".format(K, T, m))

xs = np.linspace(x[0], x[-1], 1000)
ys = funcPT1(xs, K, T, m)

poi_x = 3*T
poi_y = funcPT1(poi_x, K, T, m)
print("Slope PT1: {}".format(poi_y/poi_x))
'''


###########
# Fit PT-2
###########
K = max_y - 0.1*max_y
T = 0.01 * scale_factor_x
d = 1.05
m = 10.0 / scale_factor_x # Take this parameter from table obtained by linear regression
p0 = [K, T, d, m]

# Levenberg-Marquardt
opt_parms, parm_cov = optimize.curve_fit(funcPT2, x_fitting, y, p0, maxfev=10000)

K = opt_parms[0]
T = opt_parms[1] / scale_factor_x
d = opt_parms[2]
m = opt_parms[3] * scale_factor_x
print("K: {}, T: {}, d:{}, m: {}".format(K, T, d, m))


xs2 = np.linspace(x[0], x[-1], 100)
ys2 = funcPT2(xs2, K, T, d, m)
Ks = np.empty(100); 
Ks.fill(K)
K_on_pt2 = xs2[find_nearest_idx(ys2, K)] #xs2[np.where(ys2 >= K)[0][0]]



# Compute T
def first_derivative(t, K, T, d, m):
    T1 = T / (d + np.sqrt(d*d - 1.0))
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return -K/(T1-T2) * ( -np.exp(-t/T1) + np.exp(-t/T2) ) + m

def second_derivative(t, K, T, d, m):
    T1 = T / (d + np.sqrt(d*d - 1.0))
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return -K/(T1-T2) * ( np.exp(-t/T1)/T1 - np.exp(-t/T2)/T2 )
     
def inflection_point(T, d):
    T1 = T / (d + np.sqrt(d*d - 1.0))
    T2 = T / (d - np.sqrt(d*d - 1.0))
    return (-1/(T1-T2)) * np.log(T2/T1) * T1 * T2
  
# alternative formulation
def inflection_point2(T, d):
    root_binomic = np.sqrt((d-1)*(d+1))
    return 0.5*(np.log(1/(-1+2*np.power(d,2) - 2*root_binomic*d )) * (d-root_binomic) * T) / (root_binomic * (d-np.sqrt(np.power(d,2) - 1)))

    
        
t_inflection = inflection_point(T,d)
y_inflection = funcPT2(t_inflection, K, T, d, m) 
m_inflection = first_derivative(t_inflection, K, T, d, m)

t_intersection = (K - y_inflection) / m_inflection + t_inflection
t_dead = -y_inflection/m_inflection+t_inflection  
        
#poi_x2 = t_intersection
#poi_y2 = funcPT2(poi_x2, K, T, d, m)
#print("Slope PT2: {}".format(poi_y2/poi_x2))





############
# Plotting
###########

brewer_red = config.UIBK_blue #[0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.1, 0.1, 0.1] #[0.21568627, 0.49411765, 0.72156863]
brewer_green = config.UIBK_orange #[0.30196078, 0.68627451, 0.29019608]

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 0.48
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width

#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_small()
    
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
ax = axes

# K
ax.plot(xs2, Ks, ls='--', dashes=(3,2), linewidth=0.5, color=[0.5, 0.5, 0.5], alpha=1.0, zorder=0)

# Data
ax.plot(x, y, linewidth=0.5, color=[0.2, 0.2, 0.2], linestyle="", 
        marker='o', markeredgewidth=0.3, markersize=1.5, markeredgecolor=[0.2, 0.2, 0.2], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0, zorder=1, label='Data points')



'''
# Saturation
t_saturation_idx = find_nearest_idx(x, 0.72)
t_saturation = x[t_saturation_idx]
y_saturation = y[t_saturation_idx]
ax.annotate(r"Saturation", size=6,
            xy=(t_saturation, y_saturation), xycoords='data', 
            xytext=(0, -25), textcoords='offset points', ha="center", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )
ax.plot((t_saturation, x[-1]), (y_saturation, y_saturation+400), ls=':', dashes=(2,1), linewidth=0.5, color=[0.2, 0.2, 0.2], zorder=0, label="Hypothetical progression")
'''




#ax.plot(xs, ys, ls='-', linewidth=2.0, color=config.UIBK_blue, alpha=1.0, label="PT1 Fit")
ax.plot(xs2, ys2, ls='-', linewidth=1.0, color=config.UIBK_orange, alpha=0.75, label="Fitted function")


#ax.plot( [poi_x], [poi_y], 's', markersize=8.0, color=config.UIBK_blue, alpha=1.0, label='3T')
#ax.plot( [poi_x2], [poi_y2], '^', markersize=8.0, color=config.UIBK_orange, alpha=1.0, label='3T')
#ax.plot( [K_on_pt2], [K], '^', markersize=8.0, color=config.UIBK_orange, alpha=1.0, label='K')
#ax.plot((t_dead, t_intersection), (0, K), ls='-', linewidth=1.0, color='black', alpha=0.5) # Tangent


# K
ax.annotate("K", xy=(0.0, K), xycoords='data', fontsize=6,
                horizontalalignment='center', verticalalignment='center',
                xytext=(8, 4), textcoords='offset points')



'''
# Deformation
ax.annotate(r"Object deformation", size=6,
            xy=(1.8, 550), xycoords='data', 
            xytext=(0, -25), textcoords='offset points', ha="left", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )
'''


'''
# Flickering taxels
t_flickering_idx = find_nearest_idx(x, 0.53)
t_flickering = x[t_flickering_idx]
y_flickering = y[t_flickering_idx]
ax.annotate(r"Flickering taxels", size=6,
            xy=(t_flickering, y_flickering), xycoords='data', 
            xytext=(0.8, 80), textcoords='data', ha="left", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )

t_flickering_idx = find_nearest_idx(x, 0.56)
t_flickering = x[t_flickering_idx]
y_flickering = y[t_flickering_idx]
ax.annotate("", size=6,
            xy=(t_flickering, y_flickering), xycoords='data', 
             xytext=(0.85, 85), textcoords='data', ha="left", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )
'''


ax.set_ylim([0, 1.1*ys2.max()])

# Legend
ax.legend(loc = 'lower right', fancybox=True, shadow=False, framealpha=1.0)
ax.set_xlabel("Time [s]")
ax.set_ylabel("Mean Sensor Value", rotation=90)

fig.tight_layout()
#plt.show() 

plotname = "fitting_"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
