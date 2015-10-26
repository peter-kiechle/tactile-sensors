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
import pylab as pl



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




###########
# PT-2
###########

K = 1500
T = 0.01
d = 1.1
m = 800

xs = np.linspace(0, 0.12, 100)
ys = funcPT2(xs, K, T, d, m)
Ks = np.empty(100); 
Ks.fill(K)
K_on_pt2 = xs[np.where(ys >= K)[0][0]]
    
     
t_inflection = inflection_point(T,d)
y_inflection = funcPT2(t_inflection, K, T, d, m) 
m_inflection = first_derivative(t_inflection, K, T, d, m)

t_intersection = (K - y_inflection) / m_inflection + t_inflection
t_dead = -y_inflection/m_inflection+t_inflection


poi_x2 = t_intersection
poi_y2 = funcPT2(poi_x2, K, T, d, m)
print("Slope PT2: {}".format(poi_y2/poi_x2))





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
#figure_height = 1.3 * figure_width
figure_size = [figure_width, figure_height]

config.load_config_medium()
    
fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False, sharey=False, squeeze=True, figsize=figure_size, dpi=100)
ax = axes

ax.plot(xs, ys, ls='-', linewidth=2.0, color=config.UIBK_orange, alpha=1.0, label="PT-2 Element")
  
# auxiliary lines
ax.plot(xs, Ks, ls='--', linewidth=1.0, color=[0.5, 0.5, 0.5])
ax.plot((t_dead, t_dead), (0, K), ls=':', linewidth=1.0, color=[0.5, 0.5, 0.5])
ax.plot((t_intersection, t_intersection), (0, K), ls=':', linewidth=1.0, color=[0.5, 0.5, 0.5])

# tangent
ax.plot((t_dead, t_intersection), (0, K), ls='-', linewidth=1.0, color='black')




# Replace axis with arrows
ax.set_xlim([0, 0.12])
ax.set_ylim([0, 1700])

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

# removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    ax.spines[side].set_visible(False)
 
# removing the axis ticks
pl.xticks([]) # labels
pl.yticks([])
ax.xaxis.set_ticks_position('none') # tick markers
ax.yaxis.set_ticks_position('none')
 
x_range = xmax-xmin
y_range = ymax-ymin

aspect=figure_width/figure_height
x_offset = 0.02 * x_range * (figure_height/figure_width)
y_offset = 0.01 * y_range * (figure_width/figure_height)


# x-axis
ax.annotate(r"$t$", xy=(xmax-xmin, 0), xycoords='data', fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                xytext=(-10, -10), textcoords='offset points')
                
ax.annotate("", xy=(xmax-xmin, 0), xycoords='data', size=10,
            xytext=(xmin-x_offset, 0), textcoords='data',
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"), zorder=0
            )
            
# y-axis
ax.annotate(r"$a(t)$", xy=(0.0, ymax-ymin), xycoords='data', fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                xytext=(0, 8), textcoords='offset points')
                
ax.annotate("", xy=(0, ymax-ymin), xycoords='data', size=10,
            xytext=(0.0, ymin-y_offset), textcoords='data',
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"), zorder=0
            )

# K
ax.annotate(r"$K$", xy=(0.0, K), xycoords='data', fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                xytext=(-10, 0), textcoords='offset points')
       


        
# tau
ax.plot( [t_dead], [0], 'o', markersize=4.0, color=config.UIBK_orange, alpha=1.0, clip_on=False) 
ax.annotate(r"$\tau_{dead}$", xy=(t_dead, 0), xycoords='data', fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                xytext=(-5, -12), textcoords='offset points')
                
ax.annotate(r"$\tau$", xy=(t_dead + (t_intersection-t_dead)/2, 0), xycoords='data', fontsize=12,
                horizontalalignment='center', verticalalignment='center',
                xytext=(-5, -12), textcoords='offset points')



# Point of inflection
ax.plot( [t_inflection], [y_inflection], 'o', markersize=4.0, color=config.UIBK_orange, alpha=1.0)
ax.annotate(r"Point of inflection $t_i$", size=10,
            xy=(t_inflection, y_inflection), xycoords='data', 
            xytext=(30, 0), textcoords='offset points', ha="left", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )

# Point of intersection
ax.plot( [t_intersection], [K], 'o', markersize=4.0, color=config.UIBK_orange, alpha=1.0)
ax.annotate(r"Point of intersection $t_k$", size=10,
            xy=(t_intersection, K), xycoords='data', 
            xytext=(5, 20), textcoords='offset points', ha="left", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )

# Tangent
x_tangent = 0.8*t_intersection
y_tangent = m_inflection*(x_tangent-t_inflection) + y_inflection
ax.annotate(r"Inflectional tangent", size=10,
            xy=(x_tangent, y_tangent), xycoords='data', 
            xytext=(-5, 25), textcoords='offset points', ha="right", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )


# K_on_pt2
ax.plot( [K_on_pt2], [K], 'o', markersize=4.0, color=config.UIBK_orange, alpha=1.0)
ax.annotate(r"System Gain", size=10,
            xy=(K_on_pt2, K), xycoords='data', 
            xytext=(0, -40), textcoords='offset points', ha="center", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )



# Drift
x_drift = 0.1
y_drift = funcPT2(x_drift, K, T, d, m)
ax.annotate(r"Linear drift", size=10,
            xy=(x_drift, y_drift), xycoords='data', 
            xytext=(0, -40), textcoords='offset points', ha="center", va="center",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3", fc="black"),
            )

# Legend
#ax.legend(loc = 'lower right')
#ax.set_xlabel("Time [s]")
#ax.set_ylabel("Average Sensor Value", rotation=90)

fig.tight_layout(pad=2)
#plt.show() 

plotname = "PT2-element2"
fig.savefig(plotname+".pdf", pad_inches=0, dpi=fig.dpi) # pdf
#fig.savefig(plotname+".pgf", pad_inches=0, dpi=fig.dpi) # pgf
plt.close()

