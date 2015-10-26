# -*- coding: utf-8 -*-

##########################################
# Load configuration file (before pyplot)
##########################################
#execfile('../matplotlib/configuration.py')
import os, sys
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Custom libraries
print("CWD: " + os.getcwd() )
lib_path = os.path.abspath('../../lib')
sys.path.append(lib_path)
import framemanager_python


def find_nearest(array, value):
    idx = np.argmin(np.abs(array - value))
    return array[idx]
                                                                             
def find_nearest_idx(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx


# Load pressure profile
profileName = os.path.abspath("roughness_test_smooth_000938-001110.dsa")
frameManager = framemanager_python.FrameManagerWrapper()
frameManager.load_profile(profileName);
numFrames = frameManager.get_tsframe_count();

matrixID = 3
averages_matrix = frameManager.get_average_matrix_list(matrixID)
tsframe5 = np.copy( frameManager.get_tsframe(156, matrixID) );
texel_a = frameManager.get_texel_list(matrixID, 2, 9)
texel_b = frameManager.get_texel_list(matrixID, 3, 9)
timestamps = frameManager.get_tsframe_timestamp_list()
timestamps = (timestamps-timestamps[0]) / 1000.0 # Relative timestamps in seconds

# Trim data
start_time = 0.4
start_idx = find_nearest_idx(timestamps, start_time) 
stop_time = 1.9
stop_idx = find_nearest_idx(timestamps, stop_time) 

#x = texel_a[start_idx:stop_idx]
#t = timestamps[start_idx:stop_idx]
#t -= t[0]

x = texel_a
t = timestamps

NFFT = 32 # Window size of FFT
noverlap = 31 # Window overlapping
Fs = 73 #len(t) / (t[-1] - t[0])  # Sampling frequency (take the average)



###########
# Plotting
###########
text_width = 6.30045 # LaTeX text width in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0
size_factor = 0.75
figure_width = size_factor*text_width
figure_height = (figure_width / golden_ratio) 
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()


fig = plt.figure(figsize=figure_size, dpi=100)

gs = gridspec.GridSpec(2, 2, width_ratios=[10, 0.25]) # rows, columns
gs.update(wspace=0.05)


#-------------------------------------------------------
# First plot
ax1 = plt.subplot(gs[0,0])
ax1.plot(t, x, linestyle='-', color=[0.0, 0.0, 0.0], alpha=1.0,
         marker='o', markeredgewidth=0.5, markersize=2.5, markeredgecolor=[0.0, 0.0, 0.0], markerfacecolor=[1.0, 1.0, 1.0] )

ax1.grid(False)
ax1.set_ylabel("Raw Sensor Value", rotation=90)

#-------------------------------------------------------
# Second plot
ax2 = plt.subplot(gs[1,0], sharex=ax1)

# Pxx is the segments x freqs array of instantaneous power, freqs is
# the frequency vector, bins are the centers of the time bins in which
# the power is computed, and im is the matplotlib.image.AxesImage instance
Pxx, freqs, bins, im = plt.specgram(x, NFFT=NFFT, Fs=Fs, noverlap=noverlap, cmap=plt.cm.RdYlBu_r) 

ax2.set_ylim([freqs[1], freqs.max()])
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Frequency [Hz]")

# Add the colorbar in a seperate axis
ax3 = plt.subplot(gs[1,1])
cbar = plt.colorbar(im, cax=ax3)
cbar.solids.set_edgecolor("face")
cbar.set_label("Relative Amplitude [dB]")

#-------------------------------------------------------
# Third plot
#ax4 = plt.subplot(gs[2,0])
#ax4.psd(x, NFFT, Fs)

ax1.set_xlim(t[start_idx], t[stop_idx])

#plt.show()
plt.subplots_adjust(top=0.97, left = 0.1, bottom=0.1, right = 0.9)  # Legend on top
plt.subplots_adjust(wspace=0.2, hspace=0.2)

plotname = "frequency_analysis"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf 





NFFT = 32 # Window size of FFT
noverlap = 31 # Window overlapping
Fs = 73 #len(t) / (t[-1] - t[0])  # Sampling frequency (take the average)

fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
ax = axes

ax.psd(x, NFFT=NFFT, Fs=Fs, noverlap=noverlap)

fig.tight_layout()
#fig.show()

plotname = "frequency_analysis_psd"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf 
