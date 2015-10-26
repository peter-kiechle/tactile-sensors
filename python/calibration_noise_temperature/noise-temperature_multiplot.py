# -*- coding: utf-8 -*-

#---------------------------------------
# Temperature-Noise Relation Analysis
#---------------------------------------

# Load configuration file before pyplot
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

# Library path
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)

import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import framemanager_python

# Force reloading of external library (convenient during active development)
reload(framemanager_python)


#---------------------------------------------------------------------------------
# Configuration
brewer_red = config.UIBK_blue #[0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.1, 0.1, 0.1] #[0.21568627, 0.49411765, 0.72156863]
brewer_green = config.UIBK_orange #[0.30196078, 0.68627451, 0.29019608]

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 0.75 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_small()

num_xaxis_ticks = 5
num_yaxis_ticks = 4

#---------------------------------------------------------------------------------


#----------------------------------------
# Load raw sensor and temperature values
#----------------------------------------

profileName = os.path.abspath("temperature-noise.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);

sensor_values = np.column_stack((frameManager.get_max_matrix_list(0),
                                 frameManager.get_max_matrix_list(1),
                                 frameManager.get_max_matrix_list(2),
                                 frameManager.get_max_matrix_list(3),
                                 frameManager.get_max_matrix_list(4),
                                 frameManager.get_max_matrix_list(5) ))
                                 
#temperatures = frameManager.get_temperature_frame_list()
corresponding_temperatures = frameManager.get_corresponding_temperatures_list()

# slice of relevant axis temperatures
# i.e. (axis 1 -> matrix 0), (axis 2 -> matrix 1) ... (axis 6 -> matrix 5)
corresponding_temperatures = corresponding_temperatures[:,1:7] 


# Turn off interactive plotting
plt.ioff()

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, squeeze=True, figsize=figure_size, dpi=100)


# Subplot Labels (Matrix 0 - 5)
matrix_description = ["Finger 0: Proximal", "Finger 0: Distal", "Finger 1: Proximal", "Finger 1: Distal", "Finger 2: Proximal", "Finger 2: Distal"]

# Odd matrix IDs in first row, even IDs in second row
axismapping = [axes[1][0], axes[0][0], axes[1][1], axes[0][1], axes[1][2], axes[0][2],]

min_temp = np.min(corresponding_temperatures)
max_temp = np.max(corresponding_temperatures)

for matrixID, name in enumerate(matrix_description):

    #---------------------------------------------------------------------------------
    # Data fitting
    #---------------------------------------------------------------------------------
    x = corresponding_temperatures[:, matrixID]
    y = sensor_values[:, matrixID]

    # Since temperature does not rise lineary within the calibration timeseries the measurements are binned
    # The regression is weighted by the bin's standard deviation
    num_bins = 20
    bin_count, bin_edges = np.histogram(x, bins=num_bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    sum_y, _ = np.histogram(x, bins=num_bins, weights=y) # Sum of sensor values within bins
    sum_y_squared, _ = np.histogram(x, bins=num_bins, weights=y*y) # Sum of squared sensor values within bins
    bin_means = sum_y / bin_count
    bin_stddevs = np.sqrt(sum_y_squared/bin_count - bin_means*bin_means)

    # Fit data using a 1st order polynomial (Ordinary Least Squares)
    #[a, b], [SSE, _, _, _]  = poly.polyfit(x, y, 1, full=True, w=weights)

    # Fit data using a 1st order polynomial (Weighted Least Squares)
    # Weights are the reciprocal standard deviations 
    [a, b], [SSE, _, _, _]  = poly.polyfit(bin_centers, bin_means, 1, full=True, w=1/bin_stddevs)    
    
    
    #---------------------------------------------------------------------------------
    # Calculate RMSE band
    #---------------------------------------------------------------------------------    
    p = poly.Polynomial([a, b]) # Convenience function to evaluate polynom p(x) ≈ y

    # create series of new test x-values to predict for
    #x_new = np.linspace(np.min(x), np.max(x), num=2)
    x_new = np.linspace(min_temp, max_temp, num=2)
    y_new = p(x_new)
    
    # Sums of products of the deviations of observed values from the mean
    Sxx = sum( np.power((x-np.mean(x)), 2) )
    Syy = sum( np.power((y-np.mean(y)), 2) )
    Sxy = sum((x-np.mean(x))*(y-np.mean(y)))

    SSE = Syy-b*Sxy # Error sum of squares
    N = len(x)
    dof = N-2   # Degrees of freedom: Sample size - number of parameters
    
    MSE = SSE/dof # Mean Square Error
    
    # Standard error of estimate
    RMSE = np.sqrt(MSE);

    #MSE = SSE/dof
    #RMSE = scipy.sqrt(MSE)

    std_errors = 3    
    RMSE_lower = y_new - std_errors*RMSE
    RMSE_upper = y_new + std_errors*RMSE

    print("Regression line of matrix %d: %0.2f + %0.2f x, RMSE: %f" %(matrixID, a, b, RMSE))

    #---------------------------------------------------------------------------------
    # Plotting
    #---------------------------------------------------------------------------------

    # Select subplot
    ax = axismapping[matrixID]
    
    # Data points
    #points, = ax.plot(x, y, '.', markersize=3, 
    #                  markeredgewidth=0.0, markeredgecolor=[0.0, 0.0, 0.0],
    #                  markerfacecolor=[0.0, 0.0, 0.0], alpha=0.25) 
       
    # Errorbar
    #SEM = bin_stddevs/np.sqrt(bin_count) # Standard error of the mean
    bar = ax.errorbar(bin_centers, bin_means, yerr=bin_stddevs, 
                capsize=1.0, markersize=4, markeredgewidth=0.5, fmt='.', ecolor=[0.0, 0.0, 0.0, 1.0], 
                markeredgecolor=[0.0, 0.0, 0.0, 1.0], markerfacecolor=[1.0, 1.0, 1.0, 1.0])

    
    # Regression line
    reg_line, = ax.plot(x_new, y_new, linestyle="solid", 
                        linewidth=1.0,
                        dash_joinstyle='round',
                        dash_capstyle='round',
                        color=config.UIBK_orange, alpha=0.75)
    #reg_line.set_dashes([2, 3]) # dash, pause, dash...

    # Standard error band
    ax.fill_between(x_new, RMSE_lower, RMSE_upper, linestyle="solid", facecolor=[0.3, 0.3, 0.3], alpha=0.1)

    # Regression parameters
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
    ax.text(0.95, 0.075, r"Regression line: $%0.2f + %0.2f x$" %(a, b), 
             horizontalalignment='right',
             verticalalignment='center',
             fontsize='medium',
             transform=ax.transAxes, 
             bbox=bbox_props)
               

    # Matrix ID
    """
    ax.text(0.05, 0.915, matrix_description[matrixID], 
             horizontalalignment='left',
             verticalalignment='center',
             fontsize='medium',
             transform=ax.transAxes, 
             bbox=bbox_props)
    """
    
         
    # Axis ticks
    ax.xaxis.set_major_locator(MaxNLocator(num_xaxis_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(num_yaxis_ticks))

    ax.set_title(matrix_description[matrixID])
    #ax.set_xlim([x.min(), x.max()])
    ax.set_xlim([min_temp, max_temp])
    #ax.grid('on')
    
#---------------------------------------------------------------------------------

# Adjust margins and padding of entire plot
plt.subplots_adjust(top=0.88, left = 0.07, bottom=0.08, right = 0.98)  # Legend on top
plt.subplots_adjust(wspace=0.1, hspace=0.25)

# Set common axis labels
from matplotlib.font_manager import FontProperties
customfont = FontProperties()
customfont.set_size(mpl.rcParams['axes.titlesize'])
fig.text(0.5, 0.00, r"Temperature in corresponding axis [$\,^{\circ}\mathrm{C}$]", ha='center', va='bottom', fontproperties=customfont)
fig.text(0.00, 0.5, r"Maximum Sensor Value (Noise)", ha='left', va='center', rotation='vertical', fontproperties=customfont)



# Legend
dummy = plt.Rectangle((0, 0), 1, 1, facecolor=[0.3, 0.3, 0.3], alpha=0.1) # fill_between is not matplotlib artist compatible
plt.figlegend([bar, reg_line, dummy], [r"Sample mean ($1\sigma$)", r"Regression line", r"Prediction band: {0}-RMSE (99.7 \%)".format(std_errors)], 
              loc = 'upper center', ncol=5, labelspacing=10.0, fancybox=True, shadow=False )


#fig.suptitle("Title centered above all subplots", fontsize=14)

#plt.show()

#plotname = "noise_threshold_matrix_%d" %matrixID
plotname = "noise_threshold_linear_regression"

fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close(fig)



