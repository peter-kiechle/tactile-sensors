# -*- coding: utf-8 -*-

import os, sys

# Load configuration file before pyplot
config_path = os.path.abspath('../matplotlib/')
sys.path.append(config_path)
import configuration as config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from numpy.polynomial import Chebyshev as T



############
# Plotting
############

#orange = config.alphablend(config.UIBK_orange, 0.75)    


# For plotting
text_width = 6.30045 # LaTeX text width in inches
text_height = 9.25737 # LaTeX text height in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0



brewer_red = [0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.21568627, 0.49411765, 0.72156863]
brewer_green = [0.30196078, 0.68627451, 0.29019608]
brewer_orange = [1.0, 0.5, 0.0]
brewer_purple = [0.41568627450980394, 0.23921568627450981, 0.6039215686274509]
brewer_yellow = [1.0, 1.0, 0.2]
brewer_brown = [0.6509803921568628, 0.33725490196078434, 0.1568627450980392]
grey = [0.3, 0.3, 0.3]

colors = [brewer_red, brewer_blue, brewer_green, brewer_orange, brewer_purple, grey]
colors = map(tuple, colors)
#colors = map(lambda color : tuple(config.darken(color, 0.1)), colors)


'''
blue = [0.29803922, 0.44705882, 0.69019608]
green = [0.33333333, 0.65882353, 0.40784314]
red = [0.76862745, 0.30588235, 0.32156863]
purple = [0.50588235,  0.44705882, 0.69803922]
brown = [0.8, 0.7254902, 0.45490196]
aquamarin = [0.39215686, 0.70980392, 0.80392157]

colors = map(tuple, [blue, green, red, purple, brown, aquamarin])
'''

'''
# Brewer colormaps
import brewer2mpl
#bmap = brewer2mpl.get_map('Set2', 'qualitative', 8)
#bmap = brewer2mpl.get_map('Dark2', 'qualitative', 8)
#bmap = brewer2mpl.get_map('YlOrRd', 'sequential', 8)
bmap = brewer2mpl.get_map('Spectral', 'diverging', 6)
colors = bmap.mpl_colors
#colors = colors[::-1] # Reverse order
'''



'''
alpha = np.linspace(0.2,1,6)
colors = map(lambda a : tuple(config.alphablend(config.UIBK_blue, a)), alpha)
'''

'''
# Arbitrary colormap
import matplotlib.colors as colors
import matplotlib.cm as cmx
colorMap = plt.get_cmap("jet")
colorNorm  = colors.Normalize(vmin=0, vmax=5)
scalarMap = cmx.ScalarMappable(norm=colorNorm, cmap=colorMap)
colors = scalarMap.to_rgba(np.arange(0,6))
'''

'''
# Human friendly HSL
import husl
nColors = 6
colors = []
hues = np.linspace(0, 360-(360/nColors), nColors) # [0, 360]
saturation = 90 # [0, 100]
lightness = 65 # [0, 100]
for h in range(nColors):
    colors.append( husl.husl_to_rgb(hues[h], saturation, lightness) )
'''


'''
# Some Categorical colors
colors = [(0.12156862745098039, 0.4666666666666667, 0.7058823529411765), # Blue
          (1.0, 0.4980392156862745, 0.054901960784313725), # Orange
          (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), # Green
          (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), # Red
          (0.5803921568627451, 0.403921568627451, 0.7411764705882353), # Purple
          (0.09019607843137255, 0.7450980392156863, 0.8117647058823529), # Aquamarine
          (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), # Brown
          (0.5, 0.5, 0.5), # Gray
          (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), # Greenish/yellow
          (0.8901960784313725, 0.4666666666666667, 0.7607843137254902)]  # Pink
'''




##################################
# 1D
##################################

#figure_width = 0.35*text_width
#figure_height = 0.75*figure_width #(figure_width / golden_ratio) #figure_width

figure_width = 0.75*text_width
figure_height = (figure_width / golden_ratio) #figure_width
figure_size = [figure_width, figure_height]
config.load_config_medium()

#fig = plt.figure(figsize=figure_size, dpi=100)
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, squeeze=False, figsize=figure_size, dpi=100)
ax = axes[0][0]
x = np.linspace(-1, 1, 100)

for i in range(6):
    ax.plot(x, T.basis(i)(x), lw=1.0, color=colors[i], label="$t_%d(x)$"%i)
    #ax.plot(x, pow(x, i), lw=1.0, color=colors[i], label="$x^%d$"%i)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$t_n(x)$", rotation=90)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.xaxis.labelpad = 5
ax.yaxis.labelpad = 2

# Shrink current axis's height by 10% on the bottom
#box = ax.get_position()
#ax.set_position([box.x0, box.y0 - 0.1*box.height, box.width, 0.9*box.height]) #left, bottom, width, height

plt.subplots_adjust(top=0.85, left=0.12, bottom=0.12, right=0.98)  # Legend on top


ax.legend(loc = 'upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, fancybox=True,
                                                                   fontsize=8, 
                                                                   labelspacing=0.0, 
                                                                   handlelength=1.0, 
                                                                   handletextpad=0.2, 
                                                                   borderaxespad=0.0,
                                                                   borderpad = 0.4,
                                                                   columnspacing = 1.5)  #.get_frame().set_alpha(0.5)

#plt.tight_layout()

#plt.show()
plotname = "polynomials_chebyshev"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf

plt.close()




##################################
# 2D
##################################

figure_width = 0.66*text_width
figure_height = figure_width # (figure_width / golden_ratio) #figure_width
#figure_height = 0.75 * figure_width
figure_size = [figure_width, figure_height]
config.load_config_small()

# Custom colormap UIBK Orange
cdict = {'red': ((0.0, 1.0, 1.0),
                 (1.0, 1.0, 1.0)),

        'green': ((0.0, 1.0, 1.0),
                  (1.0, 0.5, 0.5)),

        'blue': ((0.0, 1.0, 1.0),
                 (1.0, 0.0, 0.0))}
                
plt.register_cmap(name='UIBK_ORANGES', data=cdict)

p = 5
x = np.linspace(-1, 1, 256)
y = np.linspace(-1, 1, 256)
xx, yy = np.meshgrid(x, y, sparse=False)



fig, axes = plt.subplots(nrows=p+1, ncols=p+1, sharex=True, sharey=True, squeeze=False, figsize=figure_size, dpi=300)

for row in range(p+1):
    for col in range(p+1):
        ax = axes[row][col]
        
        z = T.basis(col)(xx) * T.basis(row)(yy)

        # pcolormesh aligns cells on their edges, while imshow aligns them on their centers.
        #ax.pcolormesh(z, cmap=plt.get_cmap('afmhot_r'), vmin=-1.0, vmax=1.0)
        im = ax.imshow(z, interpolation='nearest', cmap=plt.cm.afmhot, vmin=-1.0, vmax=1.0, extent=[-1,1,-1,1]) #plt.cm.Greys_r  plt.cm.afmhot plt.cm.coolwarm
        #ax.pcolor(xx, yy, z,  cmap=plt.cm.afmhot_r)
        
        if row == 0:
            ax.set_xlabel("$t_%d$"%col)
            ax.xaxis.set_label_position('top')
            #ax.xaxis.labelpad = 5
        if col == 0:
            ax.set_ylabel("$t_%d$"%row, rotation=0)
            ax.yaxis.set_label_position('left')
            ax.yaxis.labelpad = 10

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        #ax.set_xlim(-1, 1)
        #ax.set_ylim(-1, 1)


plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.02, wspace=0.1, hspace=0.1)
#plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)


#plt.show()
plotname = "polynomials_chebyshev_2D"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()







#----------------------
# Stand-alone colorbar
#----------------------

figure_size = [0.1*figure_width, figure_height]

#fig = plt.figure(figsize=figure_size, dpi=100)
fig, axes = plt.subplots(1, 1, sharex=True, sharey=True, squeeze=False, figsize=figure_size, dpi=100)

z = T.basis(1)(xx) * T.basis(1)(yy)
im = plt.imshow(z, interpolation='nearest', cmap=plt.cm.afmhot, vmin=-1.0, vmax=1.0, extent=[-1,1,-1,1]) #plt.cm.Greys_r  plt.cm.afmhot plt.cm.coolwarm
plt.gca().set_visible(False)

# Make an axis for the colorbar on the right side
#cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])

cax = plt.axes([0.05, 0.02, 0.5, 0.90]) # (left, bottom, width height
cbar = plt.colorbar(im, cax=cax)
cbar.solids.set_edgecolor("face")
#cbar.solids.set_rasterized(True) 

#cbar.set_label('Blah')

#plt.subplots_adjust(left=0.1, right=0.5, top=0.9, bottom=0.1)
#plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

plotname = "polynomials_chebyshev_2D_colorbar"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi)
fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi)
plt.close()

