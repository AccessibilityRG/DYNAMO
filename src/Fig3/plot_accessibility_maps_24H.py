# -*- coding: utf-8 -*-
"""
plot_accessibility_maps_24H.py

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
  
  dynamo_utils.py by Digital Geography Lab / Accessibility Research Group (University of Helsinki) is licensed under
  a Creative Commons Attribution 4.0 International License.
  More information about license: http://creativecommons.org/licenses/by-sa/4.0/
"""

import geopandas as gpd
import pandas as pd
import pysal as ps
import matplotlib.colors as col
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from geopandas.plotting import plot_polygon_collection
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection
import os, matplotlib
import matplotlib as mpl
import seaborn as sns
from shapely.geometry import Polygon, MultiPolygon


# Use seaborn-whitegrid style
matplotlib.style.use('seaborn-whitegrid')

def hexColors(colormap_name, N):
    # Specify N amount of colors using colormap name (default 'jet') 
    #cmap = lineColors(colormap_name)
    
    cmap = plt.get_cmap(colormap_name, N)
        
    # Get colormap colors as Hex-colors
    hex_colors = parseHexColors(cmap)  
    #print(hex_colors)
    return cmap, hex_colors

def parseHexColors(cmap):
    hex_colors = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3] # will return rgba, we take only first 3 so we get rgb
        hex_colors.append(matplotlib.colors.rgb2hex(rgb))
    return hex_colors
    
def SeabornColors(cmap_name):
    return sns.color_palette(cmap_name)
   
def parseHexSeaborn(snsColorpalette):
    return snsColorpalette.as_hex()

def gencolor(N, colormap='Set1'):
    """
    Color generator intended to work with one of the ColorBrewer
    qualitative color scales.
    Suggested values of colormap are the following:
        Accent, Dark2, Paired, Pastel1, Pastel2, Set1, Set2, Set3
    (although any matplotlib colormap will work).
    """
    from matplotlib import cm
    # don't use more than 9 discrete colors
    n_colors = min(N, 9)
    cmap = cm.get_cmap(colormap, n_colors)
    colors = cmap(range(n_colors))
    for i in range(N):
        yield colors[i % n_colors]

        
def _flatten_multi_geoms(geoms, colors):
    """
    Returns Series like geoms and colors, except that any Multi geometries
    are split into their components and colors are repeated for all component
    in the same Multi geometry.  Maintains 1:1 matching of geometry to color.
    "Colors" are treated opaquely and so can actually contain any values.
    Returns
    -------
    components : list of geometry
    component_colors : list of whatever type `colors` contains
    """
    components, component_colors = [], []

    # precondition, so zip can't short-circuit
    assert len(geoms) == len(colors)
    for geom, color in zip(geoms, colors):
        if geom.type.startswith('Multi'):
            for poly in geom:
                components.append(poly)
                # repeat same color for all components
                component_colors.append(color)
        else:
            components.append(geom)
            component_colors.append(color)
    return components, component_colors
    
def plotCustomColors(ax, df, column, custom_cmap, linewidth=1.0, alpha=1.0, edgecolor='black'):
    
    # Get Min and max
    vmin=None
    vmax=None
    
    # Flatten possible MultiPlygons
    components, component_colors_or_values = _flatten_multi_geoms(df['geometry'], df[column])
    
    collection = PatchCollection([PolygonPatch(poly) for poly in components], linewidth=linewidth, edgecolor="black", alpha=1.0)
    collection.set_array(np.array(df[column]))
    collection.set_cmap(custom_cmap)
    collection.set_clim(vmin, vmax)
    
    ax.add_collection(collection, autolim=True)
    #ax.autoscale_view()
    #plt.draw()
    return ax

def renameTo12HourSystem(data, tcols):
    # Create dictionary for changing the names
    new_names = {}
    
    for t in tcols:
        # Get the value
        time = int(t[1:])
        
        # If time is lower than 12
        if time == 0:
            new_names[t] = '%s AM' % (time+12)
        elif time < 12:
            new_names[t] = '%s AM' % time
        elif time == 12:
            new_names[t] = '%s PM' % time
        else:
            new_names[t] = '%s PM' % (time-12)
    
    data = data.rename(columns=new_names)
    
    return data, list(new_names.values())

def renameTo24HourSystem(data, tcols, minutes=False):
    # Create dictionary for changing the names
    new_names = {}
    
    for t in tcols:
        # Get the value
        time = int(t[1:])
        
        if minutes:
            new_names[t] = str(time).zfill(2) + ":00"
        else:
            new_names[t] = str(time).zfill(2)
        
    data = data.rename(columns=new_names)
    return data, list(new_names.values())

def createDateTimeIndex(df, time_col, target_col):
    # Create datetime presentation of the open / close times
    df[target_col] = pd.to_datetime(df[time_col])
    return df

def createTimeBooleans(df, opendt, closedt, starth=0, endh=24):
    """This function creates a boolean for each hour starting from <starth> and ending to <endh>. 
    If the store is open at specified time, it will get a value 1, if not it will get 0."""
        
    # Create boolean columns indicating that a store is open at specific hours of the day
    for t in range(starth, endh): #range(0,24):
        
        # Create column
        name = 'h{0}'.format(t)
        df[name] = None
        targetdt = pd.to_datetime("%s:00" % str(t).zfill(2))
        # Set value 1 for stores that are open at that time
        for idx, row in df.iterrows():
            if targetdt >= row[opendt] and targetdt < row[closedt]:
                df.loc[idx, name] = 1
            else:
                df.loc[idx, name] = 0
    return df
    
def extractDfExterior(gdf):
    """Extracts the boundaries of given GeoDataFrame.
    
    Returns:
        A GeoDataFrame with column 'geometry' containing the exterior of the input DataFrame.
    """
    # Create Unary union
    u_union = gdf['geometry'].unary_union
    
    # Extract Shapely geometry out of the exterior of the polygon
    exterior = gpd.GeoDataFrame([[u_union.exterior]], columns=['geometry'], crs=gdf.crs)
    return exterior

def main():

    # Parameters
    # ----------
    
    show_legend = True
    show_title_text = True
                    
    # Filepaths
    data_fp = "data/Travel_time_Maps.shp"
    roads_fp = "data/Tallinn_main_roads_for_visualization.shp"
    boundaries_fp = "data/TLN_bordersDASY.shp"
    water_fp = "data/TLN_water_clip_OSM.shp"
    groceries_fp = "data/Groceries.shp"
    outdir = "results/accessibility_maps"
    
    # Read files
    data = gpd.read_file(data_fp)
    roads = gpd.read_file(roads_fp)
    boundaries = gpd.read_file(boundaries_fp)
    water = gpd.read_file(water_fp)
    shops = gpd.read_file(groceries_fp)
    
    # Re-project all into the same crs as grid
    roads['geometry'] = roads['geometry'].to_crs(crs=data.crs)
    roads.crs = data.crs
    
    boundaries['geometry'] = boundaries['geometry'].to_crs(crs=data.crs)
    boundaries.crs = data.crs
    
    water['geometry'] = water['geometry'].to_crs(crs=data.crs)
    water.crs = data.crs
    
    shops['geometry'] = shops['geometry'].to_crs(crs=data.crs)
    shops.crs = data.crs
    
    # Take only largest waterbodies
    water['area'] = water.area
    water = water.sort_values(by='area', ascending=False)
    water.reset_index(inplace=True)
    water = water.ix[0:2]
    
    # Select only primary and trunk roads
    #road_classes = ['trunk', 'primary'] #, 'primary_link', 'secondary']
    #roads = roads.ix[roads['fclass'].isin(road_classes)]
                     
    # Create Datetime Indices for opening and closing times
    shops = createDateTimeIndex(shops, time_col='open', target_col='opendt')
    shops = createDateTimeIndex(shops, time_col='close', target_col='closedt')
    
    # Create Time Booleans that shows if store is open or not (0/1 dummy-coding)
    shops = createTimeBooleans(shops, starth=0, endh=24, opendt='opendt', closedt='closedt')
    
    # Time columns showing the difference between static and interactive
    tcols = ["H%s" % num for num in range(0,24)]
    
    # Create Custom classifier (bins are the upper boundary of the class (including the value itself))
    my_bins = [5, 10, 15, 30, 45, 60]
    classifier = ps.User_Defined.make(bins=my_bins)
    
    # Classify following columns
    ccolumns= tcols
    
    classif = data[ccolumns].apply(classifier)
    
    # Rename classified column names (add letter c in front)
    classif.columns = list(map(lambda x: "c" + x, classif.columns))
    
    # Join back to grid
    data = data.join(classif)
    
    # Classified time columns showing the difference between static and interactive
    ccols = ["cH%s" % num for num in range(0,24)]
    
    # Rename columns and take the 'H' letter from the beginning away
    data, new_cols = renameTo24HourSystem(data, tcols, minutes=True)
    
    # Create a custom colormap for the map using same colors as in other plots
    # Useful info: http://basemaptutorial.readthedocs.io/en/latest/cmap.html
    # ------------------------------------------------------------------------
    
    # Plot base-grid (with no-data hashes)
    show_noData = False
    
    # ----------
    # Seaborn
    # ----------
    n=7
    # Custom Seaborn colors (however the center needs to be in custom place!)
    palette = sns.diverging_palette(220, 20, n=n)
    
    # Get hex colors
    hex_colors = parseHexSeaborn(palette)
    
    # Reverse
    hex_colors.reverse()
    
    # Plot all time-levels
    # --------------------
    
    N_colors = len(hex_colors)
    
    # Legend labels 
    legend_labels = list(my_bins)
    legend_labels.insert(0,0)
    legend_labels[-1] = "60 <"
    
    for tattribute in new_cols:
       
        # Color balancer
        color_balancer = list(hex_colors)
            
        # Print the classes
        classcol = "cH%s" % int(tattribute[0:2])
        
        # Control the number of colors    
        classes = list(data[classcol].unique())
        classes.sort()
        print("%s \t N-classes: %s \t Classes: " % (tattribute, len(classes)), classes)
    
        # If there is no values for all classes, remove the color of the specific class that is missing
        if len(classes) < N_colors:
            class_values = [val for val in range(N_colors)]
            # Put values in reverse order
            class_values.reverse()
            # Find out which classes are missing
            for i in class_values:
                if not i in classes:
                    del color_balancer[i]
        # Convert to rgb
        rgbcolors = [col.hex2color(hexcol) for hexcol in color_balancer]
    
        # Dynamo colormap
        Ncolor = len(color_balancer)
        dynamocmap = LinearSegmentedColormap.from_list("my_colormap", rgbcolors, N=Ncolor, gamma=1.0)
    
        # Initialize Figure
        if not show_legend:
            fig, ax = plt.subplots()
        else:
            fig = plt.figure(figsize=(8, 7))
            # Add axes (1 for image, 2 for custom legend)
            ax = fig.add_axes([0.05, 0.15, 0.8, 0.65])  #([DistFromLeft, DistFromBottom, Width, Height])
            ax1 = fig.add_axes([0.2, 0.08, 0.6, 0.035])
        
        # Column name for shop information
        name = "h%s" % int(tattribute[0:2])
        
        # Select only shops that are open at specified time
        selected_stores = shops.ix[shops[name]==1]
        
        if show_noData:
            # Plot base grid
            data.plot(ax=ax, color='white', linewidth = 0.1, hatch='x', edgecolor='grey')
        else:
            data.plot(ax=ax, color='white', linewidth = 0, edgecolor='grey')
        
        # Clip grid with boundaries
        data = gpd.overlay(data, boundaries, how='intersection')
        
        # Plot the map using custom color map (use the classified column)   
        ax = plotCustomColors(ax=ax, df=data, column=classcol, custom_cmap=dynamocmap, linewidth=0.05, edgecolor='grey')
        
        # Plot water bodies
        water.plot(ax=ax, color='white', alpha=1.0, linewidth=0, edgecolor='grey') #linewidth=0.05
        
        # Plot roads
        roads.plot(ax=ax, color='grey', lw=0.8, alpha=0.8)
        
        # Plot shops
        c = 'black'
        selected_stores.plot(ax=ax, color=c, marker='o', markersize=12, edgecolor=c)
        
        # Specify y and x-lim
        ax.set_xlim(left=531000, right=553000)
        ax.set_ylim(top=6596000, bottom=6579400)
        
        # Remove tick markers
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Info texts
        info_text = "%s  Stores: %s" % (tattribute, len(selected_stores))
        ppos_x = 537000; ppos_y = 6596500
        ax.text(ppos_x, ppos_y, info_text, size=30, color='black', **{'fontname':'Arial'})
        
        # Add title text
        if show_title_text:
            ax.text(ppos_x+0, ppos_y+2000, "Travel time to closest\ngrocery store in Tallinn", size=20, color='gray', **{'fontname':'Arial'})
        
        # Add legend
        if show_legend:
            
            ax1.imshow(np.arange(n).reshape(1, n), cmap=mpl.colors.ListedColormap(list(hex_colors)),
                  interpolation="nearest", aspect="auto")
            
            # Set locations of the bins
            ax1.set_xticks(np.arange(n+1) - .5)
            ax1.set_yticks([])
            
            # Specify the labels
            ax1.set_xticklabels(legend_labels)
            
            # Set colorbar title
            cbar_title = 'Travel time (minutes)'
            pos_x = 0.25; pos_y = 0.123
            plt.figtext(pos_x, pos_y, cbar_title, size=12)
        
        # Save figure
        resolution = 500
        outpath = os.path.join(outdir, "%s_Accessibility_map_%sdpi.png" % (tattribute[0:2], resolution))
        ax.axis('off')
            
        plt.savefig(outpath, dpi=resolution)
        plt.close()
        
if __name__ == "__main__":
    main()            