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


import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import scipy.optimize as optimize
from scipy import stats
import brewer2mpl

from mpl_toolkits.mplot3d import Axes3D # @UnusedImport
from matplotlib.ticker import MaxNLocator

#--------------------
# Fitting functions
#--------------------

# Proportional Time + Integral element
def funcPT1(t, K, T, m):
   if(m < 0): # ensure m > 0 by assessing a large penalty
       return 1e10
   return K * (1.0 - np.exp(-(t)/T)) + m*t

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

#---------------------------------------------------------------------------------


#--------------------
# Auxilary functions
#--------------------

def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx

def load_data_average(filename, matrixID):
    profileName = os.path.abspath(filename)
    frameManager = framemanager_python.FrameManagerWrapper()
    frameManager.load_profile(profileName);
    averages = frameManager.get_average_matrix_list(matrixID)
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    return timestamps, averages

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
    #for frameID in range(0, numFrames):
    #    centroid = np.rint(featureExtractor.compute_centroid(frameID, matrixID)).astype(int)
    #    y[frameID] = frameManager.get_texel(frameID, matrixID, centroid[0], centroid[1])
    
    # Median of nonzero values
    #numFrames = frameManager.get_tsframe_count()
    #y = np.empty([numFrames])
    #for frameID in range(0, numFrames):
    #    tsframe = frameManager.get_tsframe(frameID, matrixID)
    #    tsframe_nonzero = np.ma.masked_where(tsframe == 0, tsframe) # Restrict to nonzero values
    #    median = np.ma.median(tsframe_nonzero)
    #    y[frameID] = np.nan_to_num(median) # NaN to 0
    
    timestamps = frameManager.get_tsframe_timestamp_list()
    timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds
    
    temperatures = frameManager.get_temperature_frame_list()
    
    return timestamps, y, np.mean(temperatures[matrixID])
    
#---------------------------------------------------------------------------------


# Load Data
series = ["Room temperature", "Operating temperature", "High temperature"] 
#series = ["Room temperature"] 
profiles_A = []
profiles_B = []
profiles_C = []
profiles_D = []
profiles_E = []
profiles_F = []
profiles_G = []
profiles_H = []

profile_list = []
group_list = []

Ks_list = []
Ts_list = []
ms_list = []
xs_list = []
ys_list = []
temperature_list = []

for i, experiment in enumerate(series):
    
    # Collect involved pressure profiles
    profiles_A.append( sorted(glob.glob(series[i] + "/A*.dsa")) )
    profiles_B.append( sorted(glob.glob(series[i] + "/B*.dsa")) )
    profiles_C.append( sorted(glob.glob(series[i] + "/C*.dsa")) )
    profiles_D.append( sorted(glob.glob(series[i] + "/D*.dsa")) )
    profiles_E.append( sorted(glob.glob(series[i] + "/E*.dsa")) )
    profiles_F.append( sorted(glob.glob(series[i] + "/F*.dsa")) )
    profiles_G.append( sorted(glob.glob(series[i] + "/G*.dsa")) )
    profiles_H.append( sorted(glob.glob(series[i] + "/H*.dsa")) )
    profile_list.append(profiles_A[i] + profiles_B[i] + profiles_C[i] + profiles_D[i] + profiles_E[i] + profiles_F[i] +profiles_G[i] + profiles_H[i])
    nProfiles = len(profile_list[i])

    # Assign weight groups to profiles
    group = len(profiles_A[i]) * [17.796] 
    group += len(profiles_B[i]) * [26.694]
    group += len(profiles_C[i]) * [35.592]
    group += len(profiles_D[i]) * [44.490]
    group += len(profiles_E[i]) * [53.388]
    group += len(profiles_F[i]) * [62.286]
    group += len(profiles_G[i]) * [71.184]
    group += len(profiles_H[i]) * [80.082]    
    group_list.append(np.asarray(group))  

    Ks = np.empty([nProfiles])
    Ts = np.empty([nProfiles])
    ds = np.empty([nProfiles])
    ms = np.empty([nProfiles])
    xs = np.empty([100, nProfiles])
    ys = np.empty([100, nProfiles])
    temperatures = np.empty([nProfiles])
    for j, profile in enumerate(profile_list[i]):
        matrixID = 1
        x,y, temperature = load_smoothed(profile, matrixID);

        # Trim data
        start_idx = max(np.argmax(y!=0)-1, 0) # Index of first non-zero value (or 0)
        start_time = x[start_idx]
        stop_idx = y.shape[0]

        x = x[start_idx:stop_idx]-start_time
        y = y[start_idx:stop_idx]
        N = x.shape[0]
        print N
        # Scale data for better fitting performance
        max_y = np.max(y)
        scale_factor_x = max_y / x[-1]
        x_fitting = x*scale_factor_x # scaled version of x


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
        opt_parms, parm_cov = optimize.curve_fit(funcPT2, x_fitting, y, p0, maxfev=10000)

        K = opt_parms[0]
        T = opt_parms[1] / scale_factor_x
        d = opt_parms[2]
        m = opt_parms[3] * scale_factor_x
        # print("K: {}, T: {}, d:{}, m: {}".format(K, T, d, m))

        print i, profile, K, m 


        # Store values of single experiment
        Ks[j] = K
        Ts[j] = T
        ds[j] = d
        ms[j] = m
        xs[:,j] = np.linspace(0.0, 6.0, 100)
        ys[:,j] = funcPT2(xs[:,j], K, T, d, m)
        temperatures[j] = temperature

    # Store values of series of experiments
    Ks_list.append(Ks)
    Ts_list.append(Ts)
    ms_list.append(ms)
    xs_list.append(xs)
    ys_list.append(ys)
    temperature_list.append(temperatures)







###########
# Plotting
###########


text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 1.0
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()

# Brewer colormaps
#bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
#bmap = brewer2mpl.get_map('Dark2', 'qualitative', 8)
#bmap = brewer2mpl.get_map('YlOrRd', 'sequential', 8)
bmap = brewer2mpl.get_map('Spectral', 'diverging', 8)
colors = bmap.mpl_colors
colors = colors[::-1] # Reverse order

'''
# Arbitrary colormap
import matplotlib.colors as colors
import matplotlib.cm as cmx
colorMap = plt.get_cmap("gnuplot")
colorNorm  = plt.Normalize(vmin=0, vmax=7)
scalarMap = cmx.ScalarMappable(norm=colorNorm, cmap=colorMap)
colors = scalarMap.to_rgba(np.arange(0,8))
'''

'''
# Human friendly HSL
import husl
nColors = 8
colors = []
hues = np.linspace(0, 360-(360/nColors), nColors) # [0, 360]
saturation = 90 # [0, 100]
lightness = 65 # [0, 100]
for h in range(nColors):
    colors.append( husl.husl_to_rgb(hues[h], saturation, lightness) )
'''

'''
# Categorical colors
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # Blue
          (1.0, 0.4980392156862745, 0.054901960784313725), # Orange
          (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
          (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # Red
          (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # Purple
          (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
          (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # Greenish/yellow
          (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)] # Aquamarine
'''


# Create color and legend attributes for each category
group_colors = []
group_label = []
for i, experiment in enumerate(series):
    color_list = []
    label_list = []

    for j, entries in enumerate(profiles_A[i]):
        color_list.append(colors[0])
        if j == 0: 
            label_list.append("17.8 kPa") #200 g
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_B[i]):
        color_list.append(colors[1])
        if j == 0: 
            label_list.append("26.7 kPa") #300 g
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_C[i]):
        color_list.append(colors[2])
        if j == 0: 
            label_list.append("35.6 kPa") #400 g
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_D[i]):
        color_list.append(colors[3])
        if j == 0: 
            label_list.append("44.5 kPa") #500 g 
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_E[i]):
        color_list.append(colors[4])
        if j == 0: 
            label_list.append("53.4 kPa") #600 g 
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_F[i]):
        color_list.append(colors[5])
        if j == 0: 
            label_list.append("62.3 kPa") #700 g 
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_G[i]):
        color_list.append(colors[6])
        if j == 0: 
            label_list.append("71.1 kPa") #800 g 
        else:
            label_list.append("")

    for j, entries in enumerate(profiles_H[i]):
        color_list.append(colors[7])
        if j == 0: 
            label_list.append("80.1 kPa") #900 g 
        else:
            label_list.append("")

    group_colors.append(color_list)
    group_label.append(label_list)



# Merge testseries to combined array
temperature_flat = np.array([])
Ks_flat = np.array([])
ms_flat = np.array([])
group_flat = np.array([])
group_colors_flat = []
for i, experiment in enumerate(series):
   temperature_flat = np.concatenate((temperature_flat, temperature_list[i]))
   Ks_flat = np.concatenate((Ks_flat, Ks_list[i]))
   ms_flat = np.concatenate((ms_flat, ms_list[i]))
   group_flat = np.concatenate((group_flat, group_list[i]))
   group_colors_flat = group_colors_flat + group_colors[i]



import itertools
def polyfit2d(x, y, z, order=3, linear=False):
    """Two-dimensional polynomial fit. Based uppon code provided by 
    Joe Kington.
    
    References:
        http://stackoverflow.com/questions/7997152/python-3d-polynomial-surface-fit-order-dependent/7997925#7997925

    """
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
        if linear & (i != 0.) & (j != 0.):
            G[:, k] = 0
    m, r, _, _ = np.linalg.lstsq(G, z)
    return m, r

def polyval2d(x, y, m):
    """Values to two-dimensional polynomial fit. Based uppon code 
        provided by Joe Kington.
    """
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z










############################################
# Individual experiments (low, medium, high)
############################################
for i, experiment in enumerate(series):

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)

    for j, profile in enumerate(profile_list[i]):
        ax.plot(xs_list[i][:,j], ys_list[i][:,j], ls='-', linewidth=2.0, color=group_colors[i][j], alpha=1.0, label=group_label[i][j])

    #ax.set_title(r"Pressure calibration " + experiment + ": Fitting of $PT_2$ elements with linear drift")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Raw Sensor Value", rotation=90)
    ax.legend(loc = 'upper left', fancybox=True, shadow=False, framealpha=1.0)
    ax.set_ylim([0,1200])
    
    fig.tight_layout()
    #plt.show() 

    plotname = "pressure_calibration " + experiment
    fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
    #fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf




#################################
# 3 in 1
#################################
fig, axes = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, squeeze=True, figsize=figure_size, dpi=100)
for i, experiment in enumerate(series):
    ax = axes[i]
    for j, profile in enumerate(profile_list[i]):
       ax.plot(xs_list[i][:,j], ys_list[i][:,j], ls='-', linewidth=1.5, color=group_colors[i][j], alpha=1.0, label=group_label[i][j])
        
    ax.set_title(experiment, y=1.04)
    #ax.set_xlabel("Time [s]")
    #ax.set_ylabel("Raw Sensor Value", rotation=90)
    ax.set_ylim([0,1050])
    if i == 1:
        ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.40), ncol=4, 
                  columnspacing=2.5, labelspacing=1.0, handlelength=2.0, handletextpad=0.5, borderpad=0.5,
                  fancybox=True, shadow=False, framealpha=1.0)



# Adjust margins and padding of entire plot
plt.subplots_adjust(top=0.75, left = 0.10, bottom=0.10, right = 0.98)  # Legend on top
plt.subplots_adjust(wspace=0.1, hspace=0.0)

# Set common axis labels
from matplotlib.font_manager import FontProperties
customfont = FontProperties()
customfont.set_size(mpl.rcParams['axes.titlesize'])
fig.text(0.5, 0.00, "Time [s]", ha='center', va='bottom', fontproperties=customfont)
fig.text(0.02, 0.5, "Mean Sensor Value", ha='left', va='center', rotation='vertical', fontproperties=customfont)

  
#fig.tight_layout()
#plt.show() 

plotname = "pressure_calibration_temperature"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()



#######
# 3D K
#######

x = Ks_flat
y = temperature_flat
z = group_flat


# Fit a 2rd order, 2d polynomial
model, residuals = polyfit2d(x,y,z, 1)
for idx, val, in enumerate(model):
   print "  c_{:} &= {:.7f} \\\\".format(idx,val)
   
   
# Evaluate it on a grid...
nx, ny = 50, 50
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
zz = polyval2d(xx, yy, model)


text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.48
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 1 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.gca(projection='3d')

surf = ax.plot_surface(xx, yy, zz, rstride=7, cstride=5, 
                       color=[0.8, 0.8, 0.8], edgecolors=[0.3, 0.3, 0.3, 1.0], linewidth=0.25, shade=True, alpha=0.25,
                       antialiased=True, vmin=zz.min(), vmax=zz.max()) #cmap=plt.get_cmap('Spectral_r')
#fig.colorbar(surf)    
ax.scatter3D(x, y, z, color=group_colors_flat, s=20, edgecolors=[0.0, 0.0, 0.0]) 

# Set viewpoint.
ax.azim = 150
ax.elev = 25

# Label axes
ax.set_xlabel("Gain K")
ax.set_ylabel("Temperature [$\,^{\circ}\mathrm{C}$]", rotation=180)
ax.set_zlabel("Pressure [kPa]", rotation=-180)

#ax.axis('equal')
#ax.invert_yaxis()

ax.set_zlim([10,100])

# Background
ax.grid(False)
ax.xaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.yaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.zaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

#plt.show()

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(7))

fig.tight_layout()

plotname = "temperature_experiment_k"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()



#######
# 3D m
#######

x = ms_flat
y = temperature_flat
z = group_flat

# Fit a 2rd order, 2d polynomial
model, residuals = polyfit2d(x,y,z,2,linear=False)

for idx, val, in enumerate(model):
   print "  c_{:} &= {:.7f} \\\\".format(idx,val)

   
# Evaluate it on a grid...
nx, ny = 50, 50
xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), np.linspace(y.min(), y.max(), ny))
zz = polyval2d(xx, yy, model)

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.48
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 1 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

fig = plt.figure(figsize=figure_size, dpi=100)
ax = fig.gca(projection='3d')

#zz_m = np.ma.masked_where(zz > 100, zz)
#zz_m[np.where(np.ma.getmask(zz_m)==True)] = np.nan


# Clip values larger than coordinate box
for i in np.arange(len(xx)):
    for j in np.arange(len(yy)):
        if zz[j,i] > 100:
            zz[j,i] = 100 #np.nan
        else:
            pass
            


surf = ax.plot_surface(xx, yy, zz, rstride=7, cstride=5, #rstride=7, cstride=5, 
                       color=[0.8, 0.8, 0.8], edgecolors=[0.3, 0.3, 0.3, 1.0], linewidth=0.25, shade=True, alpha=0.25,
                       antialiased=True, vmin=zz.min(), vmax=zz.max()) # cmap=plt.get_cmap('Spectral_r')
#fig.colorbar(surf)    
ax.scatter3D(x, y, z, color=group_colors_flat, s=20, edgecolors=[0.0, 0.0, 0.0]) 

# Set viewpoint.
ax.azim = 150
ax.elev = 25

# Label axes
ax.set_xlabel("Drift m")
ax.set_ylabel("Temperature [$\,^{\circ}\mathrm{C}$]", rotation=180)
ax.set_zlabel("Pressure [kPa]", rotation=-180)

#ax.axis('equal')
#ax.invert_yaxis()

ax.set_zlim([10,100])

# Background
ax.grid(False)
ax.xaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.yaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.zaxis.pane.set_edgecolor([0.0, 0.0, 0.0, 1.0])
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

#plt.show()

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.zaxis.set_major_locator(MaxNLocator(7))

fig.tight_layout()

plotname = "temperature_experiment_m"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()




#################################
# Gain as a function of pressure
#################################

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.48
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()


# Red, Blue, Green
bmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
colors = [bmap.mpl_colors[1], # Blue
          bmap.mpl_colors[3], # Purple
          bmap.mpl_colors[0]] # Red


slope = np.zeros([len(series)])
intercept = np.zeros([len(series)])
r_value = np.zeros([len(series)])
p_value = np.zeros([len(series)])
std_err = np.zeros([len(series)])
line_xs = np.array([0.0, 100.000])

for i, experiment in enumerate(series):
    slope[i], intercept[i], r_value[i], p_value[i], std_err[i] = stats.linregress(group_list[i], Ks_list[i])


fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)

for i, experiment in enumerate(series):
   ax.plot(group_list[i], Ks_list[i], ls='', color=colors[i], 
           marker='o', markersize=2, markeredgewidth=0.3, markerfacecolor=colors[i]+(0.75,), markeredgecolor=(0, 0, 0))
   ax.plot(line_xs, slope[i]*line_xs + intercept[i], ls='-', color=colors[i], label=series[i])



#ax.set_title(r"Pressure calibration: Gain (K)")
ax.set_xlabel("Pressure [kPa]")
ax.set_ylabel("Gain K", rotation=90)
ax.legend(loc = 'upper left', fancybox=True, shadow=False, framealpha=1.0)
ax.set_xlim([15.000,85.000])
ax.set_ylim([0,1000])
#ax.set_ylim([0,4600])
fig.tight_layout()
#plt.show() 

plotname = "pressure_calibration_K"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()




###################################
# Drift as a function of pressure
##################################

slope = np.zeros([len(series)])
intercept = np.zeros([len(series)])
r_value = np.zeros([len(series)])
p_value = np.zeros([len(series)])
std_err = np.zeros([len(series)])
line_xs = np.array([0.0, 100.000])

for i, experiment in enumerate(series):
    slope[i], intercept[i], r_value[i], p_value[i], std_err[i] = stats.linregress(group_list[i], ms_list[i])


fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)

for i, experiment in enumerate(series):
   ax.plot(group_list[i], ms_list[i], ls='', color=colors[i], 
           marker='o', markersize=2, markeredgewidth=0.3, markerfacecolor=colors[i]+(0.75,), markeredgecolor=(0, 0, 0))
   ax.plot(line_xs, slope[i]*line_xs + intercept[i], ls='-', color=colors[i], label=series[i])

#ax.set_title(r"Pressure calibration: Drift (m)")
ax.set_xlabel("Pressure [kPa]")
ax.set_ylabel("Drift m", rotation=90)
ax.legend(loc = 'upper left', fancybox=True, shadow=False, framealpha=1.0)
ax.set_xlim([15.000,85.000])
ax.set_ylim([0,50])
    
fig.tight_layout()
#plt.show() 

plotname = "pressure_calibration_m"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()


#################################
# m as a function of K
#################################

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.48
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()


# Red, Blue, Green
bmap = brewer2mpl.get_map('Set1', 'qualitative', 9)
colors = [bmap.mpl_colors[1], # Blue
          bmap.mpl_colors[3], # Purple
          bmap.mpl_colors[0]] # Red


slope = np.zeros([len(series)])
intercept = np.zeros([len(series)])
r_value = np.zeros([len(series)])
p_value = np.zeros([len(series)])
std_err = np.zeros([len(series)])
line_xs = np.array([0.0, 100.000])


fig, ax = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)

for i, experiment in enumerate(series):
   ax.plot(Ks_list[i], ms_list[i], ls='', color=colors[i], 
           marker='o', markersize=2, markeredgewidth=0.3, markerfacecolor=colors[i]+(0.75,), markeredgecolor=(0, 0, 0))

ax.set_xlabel("Gain K")
ax.set_ylabel("Drift m", rotation=90)
ax.legend(loc = 'upper left', fancybox=True, shadow=False, framealpha=1.0)
#ax.set_xlim([15.000,85.000])
#ax.set_ylim([0,1000])
#ax.set_ylim([0,4600])
fig.tight_layout()
#plt.show() 

plotname = "pressure_calibration_K_vs_m"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()
