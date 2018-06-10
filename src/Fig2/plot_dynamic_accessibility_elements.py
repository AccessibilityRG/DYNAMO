# -*- coding: utf-8 -*-
"""
plot_dynamic_accessibility_elements.py

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
  This script is used to create the Figure 2 linegraphs.
  
Requirements:
  Python 3 with following packages and their dependencies: pandas, geopandas, matplotlib, numpy, seaborn, random.
  
Data:
  All data required to create the figures are in ./data/DYNvsSTAT_graphs.xlsx file. 
  
Contact:
  
  You can contact Henrikki Tenkanen if you have any questions related to the script:
     - henrikki.tenkanen (a) helsinki.fi
     - http://www.helsinki.fi/digital-geography
  
License:
  
  plot_dynamic_accessibility_elements.py by Digital Geography Lab / Accessibility Research Group (University of Helsinki) is licensed under
  a Creative Commons Attribution 4.0 International License.
  More information about license: http://creativecommons.org/licenses/by-sa/4.0/
"""

import pandas as pd
import numpy as np
from dynamo_utils import plotGraph, prePlotParams, postPlotParams, hexColors, randomValue
import matplotlib.pyplot as plt
import os

def plotAllStatic(fp, outfp, max_time, fig_width, cbar=False, colors=None):
    """Plot the line of static accessibility"""
    # Read file
    data = pd.read_excel(fp, sheet_name='ALL_STATIC', skiprows=2)
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_col = 'Population reaching food store (%)'
    xlabel = ""#acc_col
    title = ""#"All static"
    data.columns = [time_col, acc_col]
    
    # Max time
    max_time = max_time #30
    
    # Select data up to max limit
    data = data.ix[data[time_col] <= max_time]

    # Get plot parameters 
    legend, line_width, fsize, lpad, fig_width, figuresize, xticks_rot = prePlotParams(fig_width=fig_width)
    
    # Specify colors 
    colors = "black" #'#fdb063'
    
    fig, ax = plt.subplots()
    
    # Plot 
    data.plot.line(acc_col, time_col, ax=ax, stacked=True, color=colors, rot=xticks_rot, figsize=figuresize, legend=legend, lw=line_width, style="--")
    
    # Adjust image paramaters after plotting and save the figure    
    postPlotParams(outfp, ax, fsize, lpad, xlabel, title, max_time, cbar=cbar)    

def plotPopDynamic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    """Plot lines where population is dynamic but transport and activity locations are static"""
    # Read file
    data = pd.read_excel(fp, sheet_name='ONLY_POP_DYN', skiprows=2)
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_cols = ["h %s" % t for t in range(0,24)]
    title = ""#"Pop dynamic"
    xlabel = ""#"Population reaching food store (%)"
    
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, colors=colors)
        

def plotPTDynamic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    """Plot lines where transportation (PT) is dynamic but people and activity locations are static"""
    # Read file
    data = pd.read_excel(fp, sheet_name='ONLY_PT_DYN', skiprows=2)
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_cols = ["h %s" % t for t in range(0,24)]
    title = "" #"PT dynamic"
    xlabel = "" #"Population reaching food store (%)"
        
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, colors=colors)

def plotServiceDynamic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    """Plot lines where activity locations are dynamic but transport and people are static"""
    # Read file
    data = pd.read_excel(fp, sheet_name='ONLY_SERV_DYN', skiprows=2)
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    #acc_cols = ["h 7", "h 8", "h 9", "h 10-17", "h 18", "h 19", "h 20", "h 21", "h 22", "h 23-06"]
    acc_cols = ["h %s" % t for t in range(0,24)]
    title = ""#"Service dynamic"
    xlabel = ""#"Population reaching food store (%)"
    
    # Business hours (h 13 is not included because it will have the REAL values)
    bhour_cols = ["h 10", "h 11", "h 12", "h 14", "h 15", "h 16", "h 17"]
    
    # Night time hours
    nhour_cols = ["h 23", "h 0", "h 1", "h 2", "h 3", "h 4", "h 5", "h 6"]
    
    # Create some random variation for hours that have identical values throughout the busness-hours (10-17)
    for bhour in bhour_cols:
        # Get a random decimal number between -0.05 and 0.05 (five % random variation)
        random_v = randomValue(-1.5, 1.5)
        # Apply random value to create some variation for business hours
        data[bhour] = data[bhour] + random_v
        
    for nhour in nhour_cols:
        # Get a random decimal number between -0.05 and 0.05 (five % random variation)
        random_v = randomValue(-1.5, 1.5)
        # Apply random value to create some variation for night-time hours
        data[nhour] = data[nhour] + random_v
        
    fig, ax = plt.subplots()
    
    # Max time
    max_time = max_time #30
    
    # Select data up to max limit
    data = data.ix[data[time_col] <= max_time]

    # Get N amount of Hex-colors
    N = len(acc_cols)
    cmap, hex_colors = hexColors(colormap_name, N)
   
    # Get plot parameters 
    legend, line_width, fsize, lpad, fig_width, figuresize, xticks_rot = prePlotParams(fig_width=fig_width)    
    
    # Marker size
    msize=7
    
    # Plotwith custom elements (linewidths etc.)
    for idx, col in enumerate(acc_cols):
        # If time is 17 use slightly thicker solid line 
        if idx >= 10 and idx <= 17:
            data.plot.line(col, time_col, ax=ax, stacked=True, rot=xticks_rot, figsize=figuresize, legend=legend, style='-', lw=line_width-.5, color=hex_colors[idx])
        elif idx == 23 or idx <= 6:
            data.plot.line(col, time_col, ax=ax, stacked=True, rot=xticks_rot, figsize=figuresize, legend=legend, style='-', lw=line_width-.5, color=hex_colors[idx])
        # In other cases do normal plotting
        else:
            data.plot.line(col, time_col, ax=ax, stacked=True, rot=xticks_rot, figsize=figuresize, legend=legend, lw=line_width, color=hex_colors[idx])

    # Plot points for time 17 and time 23
    data.plot.line('h 17', time_col, ax=ax, stacked=True, rot=xticks_rot, figsize=figuresize, legend=legend, style='.', markersize=msize, lw=line_width+0.5, color=hex_colors[17])
    data.plot.line('h 23', time_col, ax=ax, stacked=True, rot=xticks_rot, figsize=figuresize, legend=legend, style='.', markersize=msize, lw=line_width+0.5, color=hex_colors[23])
            
    # Adjust image paramaters after plotting and save the figure    
    postPlotParams(outfp, ax, fsize, lpad, xlabel, title, max_time, N, cmap, cbar=cbar)

def plotAllDynamic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    """Plot lines where all accessibility elements (people, transport, acitivity locations) are dynamic."""
    # Read file
    data = pd.read_excel(fp, sheet_name='ALL_DYNAMIC_&_COMPARISON', skiprows=2)
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_cols = ["h %s" % t for t in range(0,24)]
    title = ""#"All dynamic"
    xlabel = ""#"Population reaching food store (%)"
            
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, colors=colors)
    
def plotAllDynamicPlusStatic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    # Read file
    data = pd.read_excel(fp, sheet_name='ALL_DYNAMIC_&_COMPARISON', skiprows=2)
    static = pd.read_excel(fp, sheet_name='ALL_STATIC', skiprows=2)
    
    # Join the files together
    data = data.merge(static, on='Time')
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_cols = ["h %s" % t for t in range(0,24)]
    title = ""#"All dynamic"
    xlabel = ""#"Population reaching food store (%)"
    
    # Column containing static information
    static_col = "h 0 - 23"
    
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, incl_static=static_col)
    
def plotSelectedAllDynamicPlusStatic(fp, outfp, max_time, fig_width, colormap_name, cbar=False, colors=None):
    """Plot lines where all accessibility elements (people, transport, acitivity locations) are dynamic. Add static line as additional element to show difference."""
    # Read file
    data = pd.read_excel(fp, sheet_name='ALL_DYNAMIC_&_COMPARISON', skiprows=2)
    static = pd.read_excel(fp, sheet_name='ALL_STATIC', skiprows=2)
    
    # Join the files together
    data = data.merge(static, on='Time')
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_cols = ["h 8", "h 13", "h 17", "h 22"]
    title = ""#"All dynamic"
    xlabel = ""#"Population reaching food store (%)"
    
    # Column containing static information
    static_col = "h 0 - 23"
    
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, incl_static=static_col, colors=colors)
    
def plotStoresSelectedAllDynamicPlusStatic(fp, outfp, store, max_time, fig_width, colormap_name, cbar=False, colors=None):
    """Plot lines for selected stores where all accessibility elements (people, transport, acitivity locations) are dynamic. Add static line as additional element to show difference."""
    # Sheet numbers in the Excel file
    stores = {
              'Lasnamäe': 'LasnamaePrisma',
              'Solaris': 'Solaris',
              'Rocca': 'RoccaPrisma'
              }
              
    # Determine the store sheet
    sheet = stores[store]
    
    # Read file
    data = pd.read_excel(fp, sheet_name=sheet, skiprows=2)
    static = pd.read_excel(fp, sheet_name='ALL_STATIC', skiprows=2)
    
    # Join the files together
    data = data.merge(static, on='Time')
    
    # Change column names ('t' --> time, 'ap' --> accessed population)
    time_col = 'Time'
    acc_cols = ["h8d", "h13d", "h17d", "h22d"]
    title = ""#"All dynamic"
    xlabel = ""#"Population reaching food store (%)"
    
    # Column containing static information
    static_col = "static"
    
    # Plot the graph
    plotGraph(data=data, acc_cols=acc_cols, time_col=time_col, title=title, xlabel=xlabel, outfp=outfp, max_time=max_time, fig_width=fig_width, colormap_name=colormap_name, cbar=cbar, incl_static=static_col, colors=colors)

# ------------
# Parameters
# ------------  

# Filepaths
fp = "/data/DYNvsSTAT_graphs.xlsx"
outfolder = "Results/Figure2"

# Specify the time limit (ylimit in graph)
max_time = 30

# Width of the figure
fig_width = 3.5 # 4.5

# Colormap
cmap_name = "Spectral"
colorbar = False

# --------------
# Create figures
# --------------

# All static
outfp = os.path.join(outfolder, "DYNAMO_elements_all_static_%sMINUTES_600dpi" % max_time)
plotAllStatic(fp, outfp, max_time, fig_width=fig_width-0, cbar=colorbar) #-2

# Population dynamic
outfp = os.path.join(outfolder, "DYNAMO_elements_pop_dynamic_%sMINUTES_600dpi.png" % max_time)
plotPopDynamic(fp, outfp, max_time, fig_width, cmap_name, cbar=colorbar)

# Services dynamic
outfp = os.path.join(outfolder, "DYNAMO_elements_service_dynamic_%sMINUTES_600dpi.png" % max_time)
plotServiceDynamic(fp, outfp, max_time, fig_width, cmap_name, cbar=colorbar)

# Public transport dynamic
outfp = os.path.join(outfolder, "DYNAMO_elements_PT_dynamic_%sMINUTES_600dpi.png" % max_time)
plotPTDynamic(fp, outfp, max_time, fig_width, cmap_name, cbar=colorbar)

# All Dynamic
outfp = os.path.join(outfolder, "DYNAMO_elements_All_dynamic_%sMINUTES_600dpi.png" % max_time)
plotAllDynamic(fp=fp, outfp=outfp, max_time=max_time, fig_width=4.5, colormap_name=cmap_name, cbar=True)

# All Dynamic + Static line
outfp = os.path.join(outfolder, "DYNAMO_elements_All_dynamic_plus_static_%sMINUTES_600dpi.png" % max_time)
plotAllDynamicPlusStatic(fp=fp, outfp=outfp, max_time=max_time, fig_width=4.5, colormap_name=cmap_name, cbar=True)

# Specify colors
# --------------
# All colors hex (colors for 24 hours)
all_hex = ['#9e0142', '#b61c48', '#ce364d', '#de4c4b', '#ec6146', '#f67848', '#f99555', '#fdb063', '#fdc675', '#fedc87', '#feeb9d', '#fff8b4', '#fafdb7', '#eff8a6', '#e1f399', '#c7e89e', '#aedea3', '#90d2a4', '#72c7a5', '#58b3ab', '#429ab6', '#3881b9', '#4b68ae', '#5e4fa2']
# Choose hex colors that will be used (8, 13, 17 22)
colors = [all_hex[7], all_hex[15], all_hex[19], all_hex[22]]

# 4 selected times to all stores: All Dynamic + Static line
# ---------------------------------------------
outfp = os.path.join(outfolder, "DYNAMO_elements_Selected_All_dynamic_plus_static_%sMINUTES_600dpi.png" % max_time)    
plotSelectedAllDynamicPlusStatic(fp=fp, outfp=outfp, max_time=max_time, fig_width=4.5, colormap_name=cmap_name, cbar=True, colors=colors)

# 4 selected times to Lasnamäe: All Dynamic + Static Line
# --------------------------------------------------------
store = "Lasnamäe"
outfp = os.path.join(outfolder, "%s_DYNAMO_elements_Selected_All_dynamic_plus_static_%sMINUTES_600dpi.png" % (store, max_time))  
plotStoresSelectedAllDynamicPlusStatic(fp=fp, outfp=outfp, store=store, max_time=max_time, fig_width=4.5, colormap_name=cmap_name, cbar=True, colors=colors)

# 4 selected times to Rocca: All Dynamic + Static Line
# --------------------------------------------------------
store = "Rocca"
outfp = os.path.join(outfolder, "%s_DYNAMO_elements_Selected_All_dynamic_plus_static_%sMINUTES_600dpi.png" % (store, max_time))
plotStoresSelectedAllDynamicPlusStatic(fp=fp, outfp=outfp, store=store, max_time=max_time, fig_width=4.5, colormap_name=cmap_name, cbar=True, colors=colors)

# 4 selected times to Solaris: All Dynamic + Static Line
# --------------------------------------------------------
store = "Solaris"
outfp = os.path.join(outfolder, "%s_DYNAMO_elements_Selected_All_dynamic_plus_static_%sMINUTES_600dpi.png" % (store, max_time))
plotStoresSelectedAllDynamicPlusStatic(fp=fp, outfp=outfp, store=store, max_time=max_time, fig_width=4.5, colormap_name=cmap_name, cbar=True, colors=colors)
