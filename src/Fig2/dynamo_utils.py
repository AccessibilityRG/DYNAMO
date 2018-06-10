# -*- coding: utf-8 -*-
"""
dynamo_utils.py

Copyright (C) 2018.  Digital Geography Lab / Accessibility Research Group, University of Helsinki (Järv, Tenkanen, Salonen Ahas & Toivonen).

Author: 
    Henrikki Tenkanen, University of Helsinki, Finland.
    
Created at:
    September 2017
    
This script is part of the following article:
  Olle Järv, Henrikki Tenkanen, Maria Salonen, Rein Ahas & Toivonen Tuuli (2018). 
  Dynamic cities: Location-based accessibility modelling as a function of time. 
  Applied Geography 95, 101-110. https://www.sciencedirect.com/science/article/pii/S014362281731144X
  
Purpose:
  This script provides generic helper functions to produce graphs shown in the article. 
  
Requirements:
  Python 3 with following packages and their dependencies: pandas, geopandas, matplotlib, numpy, seaborn, random.
  
Contact:
  
  You can contact Henrikki Tenkanen if you have any questions related to the script:
     - henrikki.tenkanen (a) helsinki.fi
     - http://www.helsinki.fi/digital-geography
  
License:
  
  dynamo_utils.py by Digital Geography Lab / Accessibility Research Group (University of Helsinki) is licensed under
  a Creative Commons Attribution 4.0 International License.
  More information about license: http://creativecommons.org/licenses/by-sa/4.0/
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import random

plt.style.use('seaborn-whitegrid')

def normalizeAccessibility(df, time_col, target_col, scale="1", max_value=None):
    """Normalizes the given accessibility times into a scale from 0.0 to 1.0
    
    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing the input dat
    time_col : pd.Series
        Pandas column containing the time values that will be normalized.
    target_col : str
        A name for the column where the normalized values will be stored.
    
    Returns
    -------
    pd.DataFrame
    
    """
    if max_value:
        if scale == "1":
            df[target_col] = df[time_col] / max_value
        elif scale == "100":
            df[target_col] = df[time_col] / max_value
        else:
            raise Exception("Scale needs to be either '1' or '100'")
    else:
        if scale == "1":
            df[target_col] = df[time_col] / df[time_col].max() 
        elif scale == "100":
            df[target_col] = df[time_col] / df[time_col].max() * 100
        else:
            raise Exception("Scale needs to be either '1' or '100'")
    return df

def getColors(N):
    # Specify the color map to be used
    cmap_name = "jet"
    cmap = plt.get_cmap(cmap_name, N)
    return cmap
    
def randomValue(min_v, max_v):
    """Produce a random decimal value between minimum and maximum values (min_v, max_v)"""
    return random.uniform(float(min_v), float(max_v))
    
def hexColors(colormap_name, N):
    """Parses colormap and hex colors from a given colormap name (e.g. 'Spectral') with N amount of colors"""
    
    cmap = plt.get_cmap(colormap_name, N)
    
    # Get colormap colors as Hex-colors
    hex_colors = parseHexColors(cmap)  
    
    return cmap, hex_colors

def parseHexColors(cmap):
    """Returns hex colors"""
    hex_colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        hex_colors.append(matplotlib.colors.rgb2hex(rgb))
    return hex_colors

def createListedColormap(color_list):
    """Creates a mpl.colors.ListedColormap from a list of colors e.g. in hex format."""
    return mcolors.ListedColormap(color_list)
    
    
def lineColors(cmap_name):
    """Add colorbar with sequences to line plot"""
    # Inspired by: http://stackoverflow.com/questions/8342549/matplotlib-add-colorbar-to-a-sequence-of-line-plots
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=plt.Normalize(vmin=0, vmax=24))
    # fake up the array of the scalar mappable. 
    sm._A = []
    return sm
    
def colorbar_index(ncolors, cmap, colors):
    """"Sets a colormap based on cmap or list of colors."""
    if not colors:
        cmap = cmap_discretize(cmap, ncolors)
    else:
        cmap = createListedColormap(colors)
        ncolors = len(colors)
        
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    colorbar.ax.tick_params(labelsize=13) 
    return mappable

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

    
def prePlotParams(legend=False, line_width=1.5, fsize=18, lpad=5, xticks_rot=0, fig_width=3.5, figuresize=None):
    """Initializes plot parameters before plotting"""
    # Show legend?
    legend = legend
    
    # Width of the bars
    line_width = line_width
    
    # Font size label
    fsize = fsize
    
    # Label padding
    lpad = lpad
    
    # Width of the figure
    fig_width=fig_width
    
    # Figsize
    if not figuresize:
        figuresize = (fig_width, 6.5)
    
    # Rotation for xticks
    xticks_rot = xticks_rot
    return legend, line_width, fsize, lpad, fig_width, figuresize, xticks_rot
    
def postPlotParams(outfp, ax, fsize, lpad, xlabel, title, max_time, N=None, cmap=None, cbar=True, colors=None):
    """Modifies plot parameters after the plot and saves the figure"""
    # Set the label size and padding
    ax.tick_params(labelsize=fsize, pad=lpad)
    
    # Set ylinm
    ax.set_ylim(top=max_time)
    ax.set_xlim(left=0, right=100.0)
    
    # Set x and y-label
    ax.set_xlabel(xlabel)
    
    # Adjust gridlines
    gridlines = ax.get_xgridlines()
    gridlines.extend( ax.get_ygridlines() )

    for line in gridlines:
        line.set_linewidth(0.5)
    ax.grid(True)
    plt.title(title)
    
    # Specify colors for the lines
    # Use discrete color classes: http://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
    if cbar:
        cbar = colorbar_index(ncolors=N, cmap=cmap, colors=colors)
        # Add text
        info_text = "Hour"
        ppos_x = 107.2; ppos_y = max_time + 0.35
        ax.text(ppos_x, ppos_y, info_text, size=16, color='black', fontstyle='normal', **{'fontname':'Arial'})
    
    # Save figure to disk
    plt.savefig(outfp, dpi=600)
    

def plotGraph(data, acc_cols, time_col, title, xlabel, outfp, max_time, fig_width, colormap_name, cbar=False, incl_static=None, colors=None):
    """Helper function to create line plots"""
    fig, ax = plt.subplots()
    
    # Max time
    max_time = max_time #30
    
    # Select data up to max limit
    data = data.ix[data[time_col] <= max_time]
    
    # Get N amount of Hex-colors
    N = len(acc_cols)
    
    # Override colormap name if 'colors' variable is used
    if not colors:
        cmap, hex_colors = hexColors(colormap_name, N)
    else:
        cmap = None
        hex_colors = colors
       
    # Get plot parameters 
    legend, line_width, fsize, lpad, fig_width, figuresize, xticks_rot = prePlotParams(fig_width=fig_width)
    
    # Plot 
    for idx, col in enumerate(acc_cols):
        data.plot.line(col, time_col, ax=ax, stacked=True, rot=xticks_rot, figsize=figuresize, legend=legend, lw=line_width, color=hex_colors[idx])
    
    # Plot static as well if needed
    if incl_static:
        data.plot.line(incl_static, time_col, ax=ax, rot=xticks_rot, legend=legend, lw=line_width, color="black", style='--')
    
    # Adjust image paramaters after plotting and save the figure    
    postPlotParams(outfp, ax, fsize, lpad, xlabel, title, max_time, N, cmap, cbar, colors=colors)
    
    
