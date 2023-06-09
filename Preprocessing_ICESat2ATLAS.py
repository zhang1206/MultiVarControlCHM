#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 16:12:56 2020

@author: zhangtianqi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import h5py
import glob
import pandas as pd
# import geopandas as gpd
import os
import os.path as osp

path = 'D:/Workfolder_Zhang/Data/CanopyHeightModel/studyRegion5/ICESat/'
os.chdir(path)

result_path = osp.join(path,'summer/')
if osp.exists(result_path) == False:
    os.makedirs(result_path) 

# for file in os.listdir(path):
#     d = os.path.join(path, file)
#     if os.path.isdir(d):
#         print(d)

#%% access all files in folders and subfolders
from os import walk

# folder path
dir_path = r'D:/Workfolder_Zhang/Data/CanopyHeightModel/studyRegion5/ICESat/summerTracks/'

# list to store files name
all_files = []
for (dir_path, dir_names, file_names) in walk(dir_path):
    for file in  os.listdir(dir_path):
        if file.endswith('.h5') and file not in all_files:
           all_files.append(os.path.join(dir_path, file)) # join subdirectory and file names
print(all_files)

#%% concatenate the canopy surface file. 
# strong beam
# all_files = glob.glob(path + "*.h5")
hght = []
for filename in all_files:
    print(filename)
    test = h5py.File(filename, 'r')
    
    if test['orbit_info']['sc_orient'][0] == 0:  
        trackNr = ['gt1l','gt2l','gt3l']
    if test['orbit_info']['sc_orient'][0] == 1:  
        trackNr = ['gt1r','gt2r','gt3r']
    if test['orbit_info']['sc_orient'][0] == 2:  
        pass
        
    DF = []
    for k in np.arange(len(trackNr)):
        # print(k)
        df = pd.DataFrame({
                    'lon': test[trackNr[k]]['land_segments']['longitude'],
                    'lat': test[trackNr[k]]['land_segments']['latitude'],
                    'SNR': test[trackNr[k]]['land_segments']['snr'],
                    'landcover': test[trackNr[k]]['land_segments']['segment_landcover'],
                    'snowcover': test[trackNr[k]]['land_segments']['segment_snowcover'],
                    'watermask': test[trackNr[k]]['land_segments']['segment_watermask'],
                    'urban_flag': test[trackNr[k]]['land_segments']['urban_flag'],
                    'terrain_h': test[trackNr[k]]['land_segments']['terrain']['h_te_interp'],
                    'canopy_h_98':test[trackNr[k]]['land_segments']['canopy']['h_canopy'],
                    'canopy_h_max':test[trackNr[k]]['land_segments']['canopy']['h_max_canopy'],
                    'canopy_h_uncertainty':test[trackNr[k]]['land_segments']['canopy']['h_canopy_uncertainty'],
                    'canopy_openness':test[trackNr[k]]['land_segments']['canopy']['canopy_openness'],
                    'toc_roughness':test[trackNr[k]]['land_segments']['canopy']['toc_roughness'],
                 })
        DF.append(df)
    DFf = pd.concat(DF, axis=0, ignore_index=True)
    hght.append(DFf)
frame = pd.concat(hght, axis=0, ignore_index=True)
frame.to_csv(result_path+'ICESat_strongBeam.csv')

#%% weak beam 
# all_files = glob.glob(path + "*.h5") 
hght = []
for filename in all_files:
    print(filename)
    test = h5py.File(filename, 'r')
    
    if test['orbit_info']['sc_orient'][0] == 0:  
        trackNr = ['gt1r','gt2r','gt3r']
    if test['orbit_info']['sc_orient'][0] == 1:  
        trackNr = ['gt1l','gt2l','gt3l']
    if test['orbit_info']['sc_orient'][0] == 2:  
        pass
    
    DF = []
    for k in np.arange(len(trackNr)):
        # print(k)
        df = pd.DataFrame({
                'lon': test[trackNr[k]]['land_segments']['longitude'],
                'lat': test[trackNr[k]]['land_segments']['latitude'],
                'SNR': test[trackNr[k]]['land_segments']['snr'],
                'landcover': test[trackNr[k]]['land_segments']['segment_landcover'],
                'snowcover': test[trackNr[k]]['land_segments']['segment_snowcover'],
                'watermask': test[trackNr[k]]['land_segments']['segment_watermask'],
                'urban_flag': test[trackNr[k]]['land_segments']['urban_flag'],
                'terrain_h': test[trackNr[k]]['land_segments']['terrain']['h_te_interp'],
                'canopy_h_98':test[trackNr[k]]['land_segments']['canopy']['h_canopy'],
                'canopy_h_max':test[trackNr[k]]['land_segments']['canopy']['h_max_canopy'],
                'canopy_h_uncertainty':test[trackNr[k]]['land_segments']['canopy']['h_canopy_uncertainty'],
                 'canopy_openness':test[trackNr[k]]['land_segments']['canopy']['canopy_openness'],
                 'toc_roughness':test[trackNr[k]]['land_segments']['canopy']['toc_roughness'],
                 })
        DF.append(df)
    DFf = pd.concat(DF, axis=0, ignore_index=True)
    hght.append(DFf)
frame = pd.concat(hght, axis=0, ignore_index=True)
frame.to_csv(result_path+'ICESat_weakBeam.csv')

#%% concatenate the canopy surface file. # all beam 
# all_files = glob.glob(path + "*.h5")
trackNr = ['gt1l','gt1r','gt2l','gt2r','gt3l','gt3r']
hght = []
for filename in all_files:
    print(filename)
    test = h5py.File(filename, 'r')
    DF = []
    for k in np.arange(len(trackNr)):
        # print(k)
        df = pd.DataFrame({
                    'lon': test[trackNr[k]]['land_segments']['longitude'],
                    'lat': test[trackNr[k]]['land_segments']['latitude'],
                    'SNR': test[trackNr[k]]['land_segments']['snr'],
                    'landcover': test[trackNr[k]]['land_segments']['segment_landcover'],
                    'snowcover': test[trackNr[k]]['land_segments']['segment_snowcover'],
                    'watermask': test[trackNr[k]]['land_segments']['segment_watermask'],
                    'urban_flag': test[trackNr[k]]['land_segments']['urban_flag'],
                    'terrain_h': test[trackNr[k]]['land_segments']['terrain']['h_te_interp'],
                    'canopy_h_98':test[trackNr[k]]['land_segments']['canopy']['h_canopy'],
                    'canopy_h_max':test[trackNr[k]]['land_segments']['canopy']['h_max_canopy'],
                    'canopy_h_uncertainty':test[trackNr[k]]['land_segments']['canopy']['h_canopy_uncertainty'],
                     'canopy_openness':test[trackNr[k]]['land_segments']['canopy']['canopy_openness'],
                     'toc_roughness':test[trackNr[k]]['land_segments']['canopy']['toc_roughness'],
                  })
        DF.append(df)
    DFf = pd.concat(DF, axis=0, ignore_index=True)
    hght.append(DFf)
frame = pd.concat(hght, axis=0, ignore_index=True)
frame.to_csv(result_path+'ICESat_allBeam.csv')



