 
# -*- coding: utf-8 -*-

# Load configuration file before pyplot
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import matplotlib.pyplot as plt
import numpy as np


# Taken from Doukeni Karakatsani's masterthesis
#x = np.array([48.5, 46.5, 45.50, 44.50, 42.50, 40.50, 38.50, 36.50, 34.50, 32.50, 30.50, 28.50, 26.50, 24.50, 22.50, 20.50, 18.50, 16.50, 14.50, 12.50, 10.50, 8.50, 6.50, 4.50, 2.50, 1.50, 0.50, 0.00])
#y = np.array([0.0, 0.3, 0.5, 1.06, 1.86, 2.12, 2.27, 1.86,1.67,1.68, 2.03, 2.00, 1.94, 1.92, 1.89, 1.88, 1.93, 2.13, 2.06, 1.86, 2.08, 1.70, 1.79, 1.18, 0.61, 0.50, 0.00, 0.00])

x = np.array([  0. ,   0.5,   1.5,   2.5,   4.5,   6.5,   8.5,  10.5,  12.5,
               14.5,  16.5,  18.5,  20.5,  22.5,  24.5,  26.5,  28.5,  30.5,
               32.5,  34.5,  36.5,  38.5,  40.5,  42.5,  44.5,  45.5,  46.5,  48.5])

y = np.array([ 0.  ,  0.  ,  0.5 ,  0.61,  1.18,  1.79,  1.7 ,  2.08,  1.86,
               2.06,  2.13,  1.93,  1.88,  1.89,  1.92,  1.94,  2.  ,  2.03,
               1.68,  1.67,  1.86,  2.27,  2.12,  1.86,  1.06,  0.5 ,  0.3 ,  0.  ])

taxel_centers = np.arange(2.15+1.7, 44.65+1, 3.4)


############
# Plotting
###########

brewer_red = config.UIBK_blue #[0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.1, 0.1, 0.1] #[0.21568627, 0.49411765, 0.72156863]
brewer_green = config.UIBK_orange #[0.30196078, 0.68627451, 0.29019608]

text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

size_factor = 1.0
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio)
#figure_height = (text_width / golden_ratio) # height is golden ratio to page width
figure_height = 0.5 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()
    
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
ax = axes

for i in range(0, len(taxel_centers)):
   ax.plot((taxel_centers[i], taxel_centers[i]), (0, 2.5), ls=':', dashes=(3,2), linewidth=0.75, color=[0.5, 0.5, 0.5])

# Data
ax.plot(x, y, linewidth=1.0, color=config.UIBK_orange, linestyle="-", 
        marker='o', markeredgewidth=0.75, markersize=4.0, markeredgecolor=[0.2, 0.2, 0.2], markerfacecolor=[1.0, 1.0, 1.0], alpha=1.0, zorder=1, label='Data points')



ax.set_xlim([0, 48.5])

#ax.set_ylim([0, 850])
#ax.set_ylim([0, 1.1*ys.max()])


# Legend
#ax.legend(loc = 'lower right')
ax.set_xlabel("y-Position on Sensor Matrix [mm]")
ax.set_ylabel(r"$\Delta$ Mean Sensor Value", rotation=90)

fig.tight_layout()
#plt.show() 

plotname = "Doukeni_Karakatsani"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
