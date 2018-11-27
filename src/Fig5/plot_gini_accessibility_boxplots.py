# -*- coding: utf-8 -*-
"""
plot_gini_accessibility_boxplots.py

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
  This script can be used to produce accessibility maps showing the travel times to closest open grocery store for all 24 hours of the day in Tallinn. 
  See Figure 3 in the article, Supplement  and GitHub repository for 24H animation
  
Requirements:
  Python 3 with following packages and their dependencies: pandas, geopandas, matplotlib, numpy, seaborn, random.
  
Contact:
  
  You can contact Henrikki Tenkanen if you have any questions related to the script:
     - henrikki.tenkanen (a) helsinki.fi
     - http://www.helsinki.fi/digital-geography
  
License:
  
  plot_gini_accessibility_boxplots.py by Digital Geography Lab / Accessibility Research Group (University of Helsinki) is licensed under
  a Creative Commons Attribution 4.0 International License.
  More information about license: http://creativecommons.org/licenses/by-sa/4.0/
"""


import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dynamo_utils import plotGraph, prePlotParams, postPlotParams, hexColors, randomValue, normalizeAccessibility
import numpy as np
import time

def dualPlotBoxBar(df_box, df_bar, outfp):
    """
    Materials
    ---------
    Good subplots docs : http://matplotlib.org/examples/pylab_examples/subplots_demo.html
    """
    
    fp = r"C:\HY-Data\HENTENKA\KOODIT\manuscripts\2017_DYNAMO\Data_for_figures\GINI_index24hours.xlsx"
    data = pd.read_excel(fp, sheetname=2)
    pop = pd.read_excel(fp, sheetname=3)
    
    # Specify columns
    pop_cols = ["H%s" % col for col in range(0,24)]
    acc_cols = ["h%s" % col for col in range(0,24)]
                
    # Get maximum accessibility value of all hours (can be used as a baseline when normalizing accessibility to 0.0-1.0 scale)
    #max_ttime = data[acc_cols].max().max()
    
    # Let's normalize accessibility within each hour, not based on 24H data 
    max_ttime = None
                
    # Calculate Gini coefficients            
    gini = calculate2DGini(df1=pop, df2=acc, cols1=pop_cols, cols2=acc_cols, max_ttime=max_ttime)
    
    # Change gini hour names to be similar than with accessibility data
    gini['h'] = gini['Hour'].str.slice(1).str.zfill(2)
    
    # Reverse Gini index (change to negative numbers so that we get the image working ... )
    gini['gini_r'] = gini['gini']*-1
        
    # Reverse Accessibility values
    data[acc_cols] = data[acc_cols]*-1
    
    # Rename columns
    newcols = [col[1:].zfill(2) for col in pop_cols]
    renamedict = {}
    for left, right in zip(pop_cols, newcols):
        renamedict[left] = right
    gini = gini.rename(columns=renamedict)    
    
    # Initialize figure and axes (sharing y-axis)
    fig, (ax1, ax2) = plt.subplots(1, 2)#, sharey=True)
    
    # Style parameters
    # ================
    
    # Texts
    # -----
    
    # Label Fontsize
    lfsize = 12
    
    # Label fontsize for hours
    hlfsize = 11
    
    # Padding for hours
    hpadding = 14.0
    
    # Patches
    # -------
    
    # Filling color: blue (color matches with earlier figures)
    fcolor = '#3881b9'
    
    # Outline color
    ocolor = 'black'
    
    # Outline width of the patches
    owidth = 0.75
    
    # Grid lines
    # ----------
    
    # Grid line width
    gwidth = 0.75
    
    # Grid line color
    gcolor = 'lightgrey'
    
    # Grid line alpha
    galpha = 0.5
    
    # Whiskers
    # --------
    
    # Exclude whisker Fliers --> Show whiskers at the maximum and minimum points (do not take outliers into account)
    wfliers = False
    
    # Whisker line color
    wcolor = 'black'
    
    # Whisker line style
    wstyle = 'dotted'
    
    # Whisker line width
    wwidth = 0.5
    
    # Whisker alpha
    walpha = 0.9
           
    # Caps
    # --------
    
    # Caps line color
    ccolor = 'black'
    
    # Caps line width
    cwidth = 0.75
    
    # Caps alpha
    calpha = 0.9
    
    # Median
    # -------
    
    # Median line color
    mcolor = 'black'
    
    # Median line width
    mwidth = 1.2
    
    # Median alpha
    malpha = 1.0
    
    # Axis 1
    # -------
        
    # Make a boxplot of the results  
    # See here how to retrieve boxplot objects in a way that you can modify them--> return type needs to be 'dict': http://stackoverflow.com/questions/19453994/styling-of-pandas-groupby-boxplots
    # See here for advanced plotting options such as colors: http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
    if wfliers:
        bp = data.boxplot(ax=ax1, column=acc_cols, vert=False, patch_artist=True, return_type='dict')
    else:
        # Draw whiskers from minimum value up to 95 % of the travel time range (exclude extreme outliers)
        bp = data.boxplot(ax=ax1, column=acc_cols, vert=False, patch_artist=True, return_type='dict', whis=[2,100]) #whis='range', showfliers=True, sym='o')

    # Box style
    # ---------
    for box in bp['boxes']:
        # change outline color
        box.set( color=ocolor, linewidth=owidth)
        # change fill color
        box.set( facecolor = fcolor )
        
    # Whisker style
    # -------------
    
    for whisker in bp['whiskers']:
        whisker.set(color=wcolor, linewidth=wwidth, linestyle=wstyle, alpha=walpha)
        
    # Caps style
    # ----------
    
    # Change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color=ccolor, linewidth=cwidth, alpha=calpha)
        
    # Median style
    # ------------
    
    ## change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color=mcolor, linewidth=mwidth, alpha=malpha)
    
    # Set xlim to 330 for accessibility data
    ax1.set_xlim(-330, 1)
    
    # Set font size of tick labels
    ax1.tick_params(labelsize=lfsize)
    
    # Set x-labels to be on every 30 minutes
    ax1.xaxis.set_ticks(np.arange(-330, 30, 30))
    
    # Draw the Figure (so that tick labels are shown properly)
    # More discussion here: http://stackoverflow.com/questions/41122923/getting-tick-labels-in-matplotlib
    fig.canvas.draw()
    
    # Hide every second label
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    # Convert to positive (remove the sign)
    labels = [l[1:] for l in labels]
    labels[-1] = '0'
    ax1.set_xticklabels(labels)
    i=0
    for l in labels[1::2]:
        labels[i] = ''
        i+=2
    ax1.set_xticklabels(labels)
     
    # Axis 2
    # -------
    
    # Make bar plot with same y columns
    gini.plot.barh(ax=ax2, x='h', y='gini_r', legend=False, color=fcolor, linewidth=owidth)
        
    # Set xlim to 1.0 for Gini coefficients
    ax2.set_xlim(0,1)
    
    # Set tixk labels to right
    #ax1.yaxis.tick_right()    
    
    # Don't show tick labels for Gini indices
    #ax1.yaxis.set_ticks([])
    
    # Don't show y label
    ax2.set_ylabel("")
    
    # Set font size of tick labels    
    ax2.tick_params(labelsize=lfsize)
    
    # Set padding different font size for hour labels
    ax2.yaxis.set_tick_params(pad=hpadding, labelsize=hlfsize)
    
    # Draw the Figure (so that tick labels are shown properly)
    # More discussion here: http://stackoverflow.com/questions/41122923/getting-tick-labels-in-matplotlib
    fig.canvas.draw()
    
    # Hide all y tick-labels from ax1
    labels = [item.get_text() for item in ax1.get_yticklabels()]
    labels = ["" for x in range(len(labels))]
    ax1.set_yticklabels(labels)
    
    # Annotate Gini values
#    for p in ax1.patches:
#        ax1.annotate(np.round(p.get_width(),decimals=2), (p.get_x() * 1.0, p.get_height() * 1.05))
    
    
    # Both axes
    # ---------
    
    # Set gridlines
    for ax in [ax1, ax2]:
        # Get x gridlines
        gridlines = ax.get_xgridlines()
        # Add y gridlines to the same list
        gridlines.extend( ax.get_ygridlines() )
        # Adjust gridlines
        for line in gridlines:
            line.set(linewidth=gwidth, color=gcolor, alpha=galpha)
        ax.grid(True)
    
    # Set texts
    title1 = "Accessibility"
    title2 = "Equity"
    xlabel1 = "Travel time (min)"
    xlabel2 = "Gini coefficient"
    
    # Set titles
    titlefont = {'fontname':'Arial', 'fontsize': '15'}
    t1 = ax1.set_title(title1, **titlefont)
    t2 = ax2.set_title(title2, **titlefont)
    # Put a bit more space between title and axis
    t1.set_y(1.01) 
    t2.set_y(1.01) 
    
    # Set x-labels
    xlabelfont = {'fontname':'Arial', 'fontsize': '13'}
    ax1.set_xlabel(xlabel1, labelpad=7, **xlabelfont)
    ax2.set_xlabel(xlabel2, labelpad=7, **xlabelfont)
        
    # Save to disk
    dpi = 600
    outfp = r"C:\HY-Data\HENTENKA\KOODIT\manuscripts\2017_DYNAMO\Figures\Gini\Gini_vs_time_variation_annotations_%s_3.png" % dpi
    plt.savefig(outfp, dpi=dpi)
    
    
def giniIndex(df, col1, col2, max_ttime=None):
    """
    Calculates gini index based on normalized population info (col1) and accessibility information (col2).
    
    See also interesting stuff from pysal: http://pysal.readthedocs.io/en/latest/library/inequality/gini.html 
    
    Parameters
    ----------
    df : pd.DataFrame
        Pandas DataFrame containing the population (normalized) and accessibility (not normalized) data on grid cells. TODO: Convert the function so that also the population data is NOT normalized!
    
    """
    
    # Name for Normalized accessibility
    ncol2 = 'n'+col2
    # Name for reversed accessibility values
    rncol2 = ncol2 + '_r'
    
    # Name for cumulative population
    cumpop = 'cumpop'
    
    # Fill travel time 0 values with 0.1
    df[col2] = df[col2].replace(to_replace={0: 0.01})
    
    # Normalize accessibility values
    df = normalizeAccessibility(df, time_col=col2, target_col=ncol2, scale="1", max_value=max_ttime)
    
    # Flip accessibility values to represent "Good accessibility: higher values - Bad accessibility: lower values"
    df[rncol2] = 1 - df[ncol2]
    
    # Sort by time and population
    df = df.sort_values(by=[rncol2, col1])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    # Calculate cumulative population percentages
    df[cumpop] = df[col1].cumsum()
    
    # Calculate bar width
    df['width'] = df[cumpop].diff()
    # Calculate bar height
    df['height'] = df[rncol2].rolling(2).sum()/2.0
    
    # Fill NaN values with zeros
    df[['width', 'height']] = df[['width', 'height']].fillna(0)
    
    # Calculate Bar area
    df['area'] = df['width'] * df['height']
    
    # Total Bar area
    tot_area = df['area'].sum()
    
    # Area
    A = 0.5 - tot_area
    
    # Gini index
    gini = A/0.5
    return df, gini
    
def calculate2DGini(df1, df2, cols1, cols2, max_ttime=None):
    # Join Datasets
    data = df1.merge(df2, on='gridcode')
    
    # DataFrame for results
    gini_indices = pd.DataFrame()    
    
    for left, right in zip(cols1, cols2):
        data, gini = giniIndex(df=data, col1=left, col2=right, max_ttime=max_ttime)
        gini_indices = gini_indices.append([[left,gini]])
    # Add column names
    gini_indices.columns = ["Hour", 'gini']
    # Reset index
    gini_indices = gini_indices.reset_index(drop=True)
    
    return gini_indices
    
    
def plotGiniAllDynamicPlusStatic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    # Read file
    data = pd.read_excel(fp, sheetname=3, skiprows=2)
    static = pd.read_excel(fp, sheetname=4, skiprows=2)
    
    # Join the files together
    data = data.merge(static, on='Time')
    
    # Normalize accessibility values
    data = normalizeAccessibility(df=data, time_col='Time', target_col='nTime')
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'nTime'
    acc_cols = ["h %s" % t for t in range(0,24)]
    title = ""#"All dynamic"
    xlabel = ""#"Population reaching food store (%)"
    
    # Column containing static information
    static_col = "h 0 - 23"
    
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, incl_static=static_col)
    return data

def main():
    
    # File paths
    fp = "data/GINI_index24hours.xlsx"
    
    outfp = "test"
    fig_width = 5.0
    cmap_name = "Spectral"
    colorbar = False
    
    # Max time as percents in this case
    max_time = 100
    
    # Specify columns
    pop_cols = ["H%s" % col for col in range(0,24)]
    acc_cols = ["h%s" % col for col in range(0,24)]
    
    pop = pd.read_excel(fp, sheetname=3)
    acc = pd.read_excel(fp, sheetname=2)
    
    # Calculate Gini coefficients            
    gini = calculate2DGini(df1=pop, df2=acc, cols1=pop_cols, cols2=acc_cols)
    
    # Plot Boxplots of travel times
    dualPlotBoxBar(df_box=acc, df_bar=None, outfp=outfp)
    
if __name__ == "__main__":
    main()