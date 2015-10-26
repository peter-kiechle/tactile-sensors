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
import DenavitHartenberg as DH

# Force reloading of libraries (convenient during active development)
#reload(DH)


UIBK_blue = [0.0, 0.1765, 0.4392]
UIBK_orange = [1.0, 0.5, 0.0]



def project_active_cells_proximal(tsframe, T02):
    points = np.empty([0,3])
    for y in range(14):
        for x in range(6):
            if(tsframe[y,x] > 0.0):
                vertex = DH.get_xyz_proximal(x, y, T02)
                points = np.vstack((points, vertex))
    return points       


def project_active_cells_distal(tsframe, T03):
    points = np.empty([0,3])
    for y in range(13):
        for x in range(6):
            if(tsframe[y,x] > 0.0):
                vertex = DH.get_xyz_distal(x, y, T03)
                points = np.vstack((points, vertex))
    return points   




def preshape_pinch(close_ratio):
    Phi0 =  90 # Rotational axis (Finger 0 + 2)
    # Finger 0
    Phi1 = -72 + close_ratio * 82
    Phi2 =  72 - close_ratio * 82
    # Finger 1
    Phi3 = -90
    Phi4 =  0
    # Finger 2
    Phi5 = -72 + close_ratio * 82
    Phi6 =  72 - close_ratio * 82
    
    return Phi0, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6



x = 2.5
y = 5
distance = []
Phi1_list = []
Phi2_list = []
Phi5_list = []
Phi6_list = []
close_ratios =  np.linspace(0, 1, 20)

for close_ratio in close_ratios:
 
    Phi0, Phi1, Phi2, Phi3, Phi4, Phi5, Phi6 = preshape_pinch(close_ratio) # Simulated grasp

    Phi1_list.append(Phi1)
    Phi2_list.append(Phi2)
    Phi5_list.append(Phi5)
    Phi6_list.append(Phi6)

    # Compute transformation matrices 
    T01_f0, T02_f0, T03_f0 = DH.create_transformation_matrices_f0(Phi0, Phi1, Phi2) # Finger 0
    T01_f2, T02_f2, T03_f2 = DH.create_transformation_matrices_f2(Phi0, Phi5, Phi6) # Finger 2

    # DH-Transform: finger 0
    P_dist = DH.create_P_dist(y)
    T_total = T03_f0.dot(P_dist)
    p = np.array([y, 0.0, x, 1.0])
    xyz_0 = T_total.dot(p)[0:3] # remove w

    # DH-Transform: finger 2
    P_dist = DH.create_P_dist(y)
    T_total = T03_f2.dot(P_dist)
    p = np.array([y, 0.0, x, 1.0])
    xyz_2 = T_total.dot(p)[0:3] # remove w

    # Distance between specified points on finger 0 and 2
    distance.append( np.sqrt(np.sum((xyz_0-xyz_2)**2)) )


distance_list = np.array(distance)
distance_diff = np.absolute(np.diff(distance_list))
Phi1_diff = np.diff(np.array(Phi1_list))
Phi2_diff = np.diff(np.array(Phi2_list))
Phi5_diff = np.diff(np.array(Phi5_list))
Phi6_diff = np.diff(np.array(Phi6_list))

combined_diff = np.vstack([Phi1_diff, Phi2_diff, Phi5_diff, Phi6_diff])
angle_diff = np.max(combined_diff, axis=0)
angular_velocity = 10 # degree / second
time_steps = angle_diff / angular_velocity
time = np.hstack([ np.array([0]), np.cumsum(angle_diff) ]) / angular_velocity

velocity_distance = distance_diff / time_steps
velocity_distance = np.hstack([velocity_distance[0], velocity_distance ])





############
# Plotting
###########
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
#figure_height = (figure_width / golden_ratio)
figure_height = 1.0 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()

#---------------------------------------------------------


fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size)

#--------------------------------------------------------------------------
ax = axes[0]
ax.plot(close_ratios[1:-1], distance[1:-1], linestyle="-", color=config.UIBK_orange, alpha=1.0, label="Distance",
         marker='o', markeredgewidth=0.75, markersize=3.0, markeredgecolor=config.UIBK_orange, markerfacecolor=[1.0, 1.0, 1.0] )

ax.set_xlabel("Close ratio")
ax.set_ylabel(r"Distance [mm]", rotation=90)

#.set_xlim([0, 2.5])
#ax.set_ylim([0, 850])

# Legend
ax.legend(loc = 'upper right', fancybox=True, shadow=False, framealpha=1.0)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
ax = axes[1]
ax.plot(time[1:-1], distance[1:-1], linestyle="-", color=config.UIBK_orange, alpha=1.0, label="Distance",
         marker='o', markeredgewidth=0.75, markersize=3.0, markeredgecolor=config.UIBK_orange, markerfacecolor=[1.0, 1.0, 1.0] )

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Distance [mm]", rotation=90)

ax.set_xlim([0, time[-1]])
#ax.set_ylim([0, 850])

# Legend
ax.legend(loc = 'upper right', fancybox=True, shadow=False, framealpha=1.0)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
ax = axes[2]
ax.plot(time[1:-1], velocity_distance[1:-1], linestyle="-", color=config.UIBK_orange, alpha=1.0, label="Velocity",
         marker='o', markeredgewidth=0.75, markersize=3.0, markeredgecolor=config.UIBK_orange, markerfacecolor=[1.0, 1.0, 1.0] )

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"Velocity [mm / s]", rotation=90)

ax.set_xlim([0, time[-1]])
#ax.set_ylim([0, 850])

# Legend
ax.legend(loc = 'lower right', fancybox=True, shadow=False, framealpha=1.0)
#--------------------------------------------------------------------------


fig.tight_layout()
#plt.show()

plotname = "grasp_preshape_close_ratio"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf