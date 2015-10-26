# -*- coding: utf-8 -*-

#---------------------------------------
# Temperature-Noise Relation Analysis
#---------------------------------------

# Library path
import os, sys
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)

import numpy as np
import numpy.polynomial.polynomial as poly

import framemanager_python

# Force reloading of external library (convenient during active development)
reload(framemanager_python)


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


# Subplot Labels (Matrix 0 - 5)
matrix_description = ["Finger 0: Proximal", "Finger 0: Distal", "Finger 1: Proximal", "Finger 1: Distal", "Finger 2: Proximal", "Finger 2: Distal"]


print("Weighted linear regression:")

for matrixID, name in enumerate(matrix_description):

    #---------------------------------------------------------------------------------
    # Data fitting
    #---------------------------------------------------------------------------------
    x = corresponding_temperatures[:, matrixID]
    y = sensor_values[:, matrixID]

    # Temperature does not rise lineary within the calibration timeseries.
    # That means the number of measurements changes for different temperature ranges
    # Therefore the measurements are binned and the regression is weighted by the bin's standard deviation
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
    x_new = np.linspace(np.min(x), np.max(x), num=2)
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


    print("Matrix %d (%-18s): %0.2f x + %0.2f, RMSE: %f" %(matrixID, name, b, a, RMSE))
