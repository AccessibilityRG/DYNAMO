# -*- coding: utf-8 -*-
"""
plot_dynamic_population_24H.py

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
  
  plot_dynamic_population_24H.py by Digital Geography Lab / Accessibility Research Group (University of Helsinki) is licensed under
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
import os, matplotlib, sys
import matplotlib as mpl
import seaborn as sns
from shapely.geometry import Polygon, MultiPolygon

# Use seaborn-whitegrid style
matplotlib.style.use('seaborn-whitegrid')

def hexColors(colormap_name, N):
    """
    Returns N number of hex color codes based on the colormap name.
    
    Parameters
    ----------
    colormap_name : str
        Name of the colormap (e.g. 'Blues')
    N : int
        Number of colors that will be returned.
        
    Returns
    -------
    Colormap (cmap) and a list of hex color codes.
    
    """
    cmap = plt.get_cmap(colormap_name, N)
    hex_colors = parseHexColors(cmap)  
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
    
    collection = PatchCollection([PolygonPatch(poly) for poly in components], linewidth=linewidth, edgecolor=edgecolor, alpha=alpha)
    collection.set_array(np.array(df[column]))
    collection.set_cmap(custom_cmap)
    collection.set_clim(vmin, vmax)
    ax.add_collection(collection, autolim=True)
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

def stackColumnValues(df, columns):
    """Takes values from multiple DataFrame columns and returns a series of all values in those columns."""
    values = []
    for col in columns:
        vals = df[col].values
        values+=list(vals)
    return pd.Series(values)             
    
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
    
    show_legend = False #True
    show_title_text = False #True
    
    # Plot base-grid (with no-data hashes)
    show_noData = False
    
    # Choose classifier (if None, use self specified classification)
    #'NaturalBreaks' #'JenksCaspall' #'MaximumBreaks' #'FisherJenks' #"HeadTail"
    mapclassifier = None 
       
    # Filepaths
    data_fp = "data/MFD_Population_24H_Tallinn_500m_grid.shp"
    roads_fp = "data/Tallinn_main_roads_for_visualization.shp"
    boundaries_fp = "data/TLN_bordersDASY.shp"
    water_fp = "data/TLN_water_clip_OSM.shp"
    outdir = "results/population_maps"
    
    # Read files
    data = gpd.read_file(data_fp)
    roads = gpd.read_file(roads_fp)
    boundaries = gpd.read_file(boundaries_fp)
    water = gpd.read_file(water_fp)
    
    # Re-project all into the same crs as grid
    roads['geometry'] = roads['geometry'].to_crs(crs=data.crs)
    roads.crs = data.crs
    
    boundaries['geometry'] = boundaries['geometry'].to_crs(crs=data.crs)
    boundaries.crs = data.crs
    
    water['geometry'] = water['geometry'].to_crs(crs=data.crs)
    water.crs = data.crs
    
    # Take only largest waterbodies
    water['area'] = water.area
    water = water.sort_values(by='area', ascending=False)
    water.reset_index(inplace=True)
    water = water.ix[0:2]
    
    # Time columns showing the share of population at different hours
    tcols = ["H%s" % num for num in range(0,24)]
    
    # Multiply by 100 to get them into percentage (0-100 representation)
    data[tcols] = data[tcols]*100
    
    # Create Custom classifier 
    # bins are the upper boundary of the class (including the value itself)
    # ---------------------------------------------------------------------
    
    # Natural Breaks classification (7 classes) that has been rounded (to have a more intuitive legend)
    my_bins = [0.05, 0.10, 0.20, 0.40, 0.80, 1.6, 3.97]
    
    # Classify following columns
    ccolumns= tcols
    
    if mapclassifier:
    
        # Stack all values
        stacked_values = stackColumnValues(df=data, columns=ccolumns)
        
        
        # Classify values based on specific classifier
        n=7
        my_bins = [x for x in range(n)]
        
        if mapclassifier == 'HeadTail':
            classif = ps.esda.mapclassify.HeadTail_Breaks(stacked_values)
        elif mapclassifier == 'FisherJenks':
            classif = ps.Fisher_Jenks(stacked_values, k=n)
        elif mapclassifier == 'NaturalBreaks':
            classif = ps.Natural_Breaks(stacked_values, k=n)
        elif mapclassifier == 'MaximumBreaks':
            classif = ps.Maximum_Breaks(stacked_values, k=n)
        elif mapclassifier == 'JenksCaspall':
            classif = ps.Jenks_Caspall(stacked_values, k=n)
        
        # Get bins
        my_bins = list(classif.bins)
    
    # Apply the chosen classification
    classifier = ps.User_Defined.make(bins=my_bins)
    classif = data[ccolumns].apply(classifier)
    
    # Rename classified column names (add letter c in front)
    classif.columns = list(map(lambda x: "c" + x, classif.columns))
    
    # Join back to grid
    data = data.join(classif)
    
    # Classified columns showing the distribution of the population
    ccols = ["cH%s" % num for num in range(0,24)]
    
    # Rename columns and take the 'H' letter from the beginning away
    data, new_cols = renameTo24HourSystem(data, tcols, minutes=True)
    
    # Select color palette
    palette = sns.diverging_palette(220, 20, n=len(my_bins))
    
    # Get hex colors
    hex_colors = parseHexSeaborn(palette)
    
    # Change White color into more reddish
    hex_colors[3] = '#FFF2F2'
    
    N = len(hex_colors)
    
    # Convert to rgb
    legendcolors = [col.hex2color(hexcol) for hexcol in hex_colors]
    
    # Legend labels
    binlabels = np.array(my_bins)
    rbinlabels = binlabels.round(2)
    legend_labels = list(rbinlabels)
    legend_labels.insert(0,0)
    
    for tattribute in new_cols:
            
        # Color balancer
        color_balancer = list(hex_colors)
            
        # Print the classes
        classcol = "cH%s" % int(tattribute[0:2])
        classes = list(data[classcol].unique())
        classes.sort()
        
        print("%s \t N-classes: %s \t Classes: " % (tattribute, len(classes)), classes)
    
        # If there is no values for all classes, remove the color of the specific 
        # class that is missing (so that coloring scheme is identical for all times)
        if len(classes) < N:
            class_values = [val for val in range(N)]
            # Put values in reverse order
            class_values.reverse()
            # Find out which classes are missing and remove the color
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
        
        if show_noData:
            # Plot base grid
            if show_legend:
                data.plot(ax=ax, color='white', linewidth = 0.1, hatch='x', edgecolor='grey', legend=True)
            else:
                data.plot(ax=ax, color='white', linewidth = 0.1, hatch='x', edgecolor='grey')
        else:
            if show_legend:
                data.plot(ax=ax, color='white', linewidth = 0, edgecolor='grey', legend=True)
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
        
        # Specify y and x-lim
        ax.set_xlim(left=531000, right=553000)
        ax.set_ylim(top=6596000, bottom=6579400)
        
        # Remove tick markers
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        
        # Info texts
        info_text = "%s" % (tattribute)
        if not show_legend:
            ppos_x = 540000; ppos_y = 6595500
        else:
            ppos_x = 540000; ppos_y = 6596500
        # Add text about time
        ax.text(ppos_x, ppos_y, info_text, size=30, color='black', **{'fontname':'Arial'})
        # Add title text
        if show_title_text:
            ax.text(ppos_x-5000, ppos_y+2000, "Population distribution in Tallinn\n   based on mobile phone data", size=20, color='gray', **{'fontname':'Arial'})
        
        # Add legend
        if show_legend:
            ax1.imshow(np.arange(N).reshape(1, N), cmap=mpl.colors.ListedColormap(list(legendcolors)),
                  interpolation="nearest", aspect="auto")
            
            # Set locations of the bins
            ax1.set_xticks(np.arange(N+1) - .5)
            ax1.set_yticks([])
            
            # Specify the labels
            ax1.set_xticklabels(legend_labels)
            
            # Set colorbar title
            cbar_title = 'Share of population (%)'
            pos_x = 0.25; pos_y = 0.123
            plt.figtext(pos_x, pos_y, cbar_title, size=12)
        
        # Save figure
        resolution = 500
        outpath = os.path.join(outdir, "%s_PopulationDistribution_map_%sdpi.png" % (tattribute[0:2], resolution))
               
        # Don't show axis borders
        ax.axis('off')
        
        if not show_legend:
            plt.tight_layout()
            
        plt.savefig(outpath, dpi=resolution)
        #plt.show()
        plt.close()
        #break
            
        
if __name__ == "__main__":
    main()        