import numpy as np
import matplotlib as mpl
from matplotlib import rc_file
import os, inspect

import colorsys
 
def brighten(color, amount):
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    value += amount
    if value > 1:
        value = 1.0
    return colorsys.hsv_to_rgb(hue, saturation, value)


def darken(color, amount):
    hue, saturation, value = colorsys.rgb_to_hsv(*color)
    value -= amount
    if value < 0:
        value = 0.0
    return colorsys.hsv_to_rgb(hue, saturation, value)


def alphablend(color, alpha, background=np.array([1.0, 1.0, 1.0]) ) :
    result = (1.0-alpha)*background + alpha*np.asarray(color)
    return result.tolist()



#######################
# Use PGF/TikZ backend
#######################
mpl.use("pgf", warn=False) # Load *before* pylab, matplotlib.pyplot

pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": [
         r"\usepackage[utf8x]{inputenc}",
         r"\usepackage[T1]{fontenc}",
         r"\usepackage{amsmath}",
         r"\usepackage{amssymb}",
         r"\usepackage{wasysym}",
         ]
}

mpl.rcParams.update(pgf_with_pdflatex)
#---------------------------------------------------------------------------------



text_width = 6.30045 # LaTeX text width in inches
text_height = 9.25737 # LaTeX text height in inches
golden_ratio = (1 + np.sqrt(5) ) / 2.0

brewer_red = [0.89411765, 0.10196078, 0.10980392]
brewer_blue = [0.21568627, 0.49411765, 0.72156863]
brewer_green = [0.30196078, 0.68627451, 0.29019608]

UIBK_blue = [0.0, 0.1765, 0.4392]
UIBK_orange = [1.0, 0.5, 0.0]


# http://matplotlib.org/users/customizing.html
# mpl.matplotlib_fname() # Get current config-file


def load_config_small():
    # Import settings
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    rc_file(filepath+'/matplotlibrc_small')
    
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    
    # Fonts
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    mpl.rcParams['font.size'] = 6 # Basis for relative fonts
    mpl.rcParams['axes.titlesize'] = 8
    mpl.rcParams['axes.labelsize'] = 6
    mpl.rcParams['xtick.labelsize'] = 5
    mpl.rcParams['ytick.labelsize'] = 5
    mpl.rcParams['legend.fontsize'] = 6
    
    # Miscellaneous
    mpl.rcParams['legend.numpoints'] = 1


def load_config_medium():
    # Import settings
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    rc_file(filepath+'/matplotlibrc_medium')
    
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    
    # Fonts
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    mpl.rcParams['font.size'] = 9 # Basis for relative fonts
    mpl.rcParams['axes.titlesize'] = 11
    mpl.rcParams['axes.labelsize'] = 9
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['legend.fontsize'] = 9
    
    # Miscellaneous
    mpl.rcParams['legend.numpoints'] = 1    


def load_config_large():
    # Import settings
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    filepath = os.path.dirname(os.path.abspath(filename))
    rc_file(filepath+'/matplotlibrc_large')
    
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    
    # Fonts
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'Computer Modern'
    mpl.rcParams['font.size'] = 11 # Basis for relative fonts
    mpl.rcParams['axes.titlesize'] = 14
    mpl.rcParams['axes.labelsize'] = 11
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    
    # Miscellaneous
    mpl.rcParams['legend.numpoints'] = 1  