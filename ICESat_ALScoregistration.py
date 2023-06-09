#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 12:21:04 2021

@author: zhangtianqi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from skimage import io
import os
import os.path as osp
import rasterio
import pandas as pd
import geopandas as gpd

path = 'D:/Workfolder_Zhang/Data/CanopyHeightModel/studyRegion1/'
# use your path
os.chdir(path)

result_path = path+'results_studyRegion1_allseason/'
if osp.exists(result_path) == False:
    os.makedirs(result_path)   
    
dataDir = 'satellite_imagery/'
ICESatDr = 'ICESat/allseason/'

#%% load DSM data for reference
DSM = rasterio.open(path + dataDir +'elevation.tif') # 
S2_NDVIstdS3 = io.imread(path + dataDir + 'S2_NDVIstdS3.tif')

#% spatial homogeneity
# S2_NDVI =io.imread(path + dataDir + 'Sentinel/S2_NDVI_median.tif.tif')

# from scipy import ndimage
# S2_NDVIstdS3 = ndimage.generic_filter(S2_NDVI, np.var, size=3)


# #%% Load LVIS data
# ALSPts = gpd.read_file(path+dataDir+ICESatDr+'LVIS_selected_proj.shp').drop(['geometry'], axis = 1)
# ALSXcoord = ALSPts['xcoord'].tolist()
# ALSYcoord = ALSPts['ycoord'].tolist()

# # retrieve the corresponding row and col index of ICESat pts
# ALSRow,ALSCol = DSM.index(ALSXcoord,ALSYcoord)
# ALSPts['RowIdx'] = ALSRow
# ALSPts['ColumnIdx'] = ALSCol

# write raster
# CCRaster = rasterio.open(path + dataDir + 'Sentinel/S2_NDVIstdS3','w',driver='Gtiff',
#                           width=DSM.width, 
#                           height = DSM.height, 
#                           count=1, crs=DSM.crs, 
#                           transform=DSM.transform, 
#                           dtype='float64')
# CCRaster.write(S2_NDVIstdS3,1)
# CCRaster.close()

#%% load ALS data
# CHMmean = io.imread(path + dataDir +'previousData/ALS_CHM_30_mean_merged.tif')
CHMrh = io.imread(path + dataDir +'CHM_RH.tif') # RH95

#%% load ICESat data
beam = 'strong'
ICESatPts = gpd.read_file(path+ICESatDr+'ICESat_'+beam +'_proj2.shp')
ICESatPts = ICESatPts.drop(['geometry'], axis = 1)

# urban flag: remove urban area and invalid CH
# Flag_nonUrban = ICESatPts['urban_flag']==0
# Flag_validCH = ICESatPts['canopy_h_9']<1e6
# Flag_valid = Flag_nonUrban & Flag_validCH
# ICESatPts = ICESatPts[Flag_valid]

ICESatPtsXcoord = ICESatPts['xcoord'].tolist()
ICESatPtsYcoord = ICESatPts['ycoord'].tolist()

# retrieve the corresponding row and col index of ICESat pts
ICESatRow,ICESatCol = DSM.index(ICESatPtsXcoord,ICESatPtsYcoord)
ICESatPts['RowIdx'] = ICESatRow
ICESatPts['ColumnIdx'] = ICESatCol

flag_validCH = (ICESatPts['canopy_h_m']<40)&(ICESatPts['canopy_h_m'] >3)
RHcanopy = ICESatPts[flag_validCH][['RowIdx','ColumnIdx','canopy_h_9']]

# add ALS data
R_RH, C_RH = RHcanopy["RowIdx"],RHcanopy["ColumnIdx"]
# RHcanopy['CHMmean'] = CHMmean[R_RH, C_RH]
RHcanopy['RH50'] = CHMrh[:,:,0][R_RH, C_RH]
RHcanopy['RH75'] = CHMrh[:,:,1][R_RH, C_RH]
RHcanopy['RH90'] = CHMrh[:,:,2][R_RH, C_RH]
RHcanopy['RH95'] = CHMrh[:,:,3][R_RH, C_RH]
RHcanopy['RH98'] = CHMrh[:,:,4][R_RH, C_RH]
RHcanopy['RH100'] = CHMrh[:,:,5][R_RH, C_RH]
RHcanopy['NDVIstdS'] = S2_NDVIstdS3[R_RH, C_RH]

RHcanopy = RHcanopy.dropna()
print(RHcanopy.shape[0])

flag = RHcanopy['NDVIstdS'] > 0.03
RHcanopy = RHcanopy[~flag].dropna()
print(RHcanopy.shape[0])

#%% comparing ICESat-2 with ALS without preprocessing -----
# Correlation
testTables = RHcanopy[['canopy_h_9','RH50','RH75','RH90','RH95','RH98','RH100']]
flag = (RHcanopy['RH50']<=0) | (RHcanopy['RH50']>100)
testTables = testTables[~flag]
testTables_corr_pearson = pd.DataFrame(testTables.corr(method='pearson').iloc[0,1:])
print(testTables_corr_pearson )
testTables_corr_pearson.rename(columns={'canopy_h_9':'R'}, inplace=True)

# RMSE
from sklearn import metrics
from sklearn.linear_model import LinearRegression

def RMSEdf(x,y):
    reg = LinearRegression().fit(x[:,np.newaxis], y)
    pred = reg.predict(x[:,np.newaxis])
    RMSE = round(np.sqrt(metrics.mean_squared_error(pred,y)),2)
    return round(RMSE,2)

testTables_RMSE = np.empty(testTables_corr_pearson.shape)
testTables_rRMSE = np.empty(testTables_corr_pearson.shape)

Rows = ['RH50','RH75','RH90','RH95','RH98','RH100']
for i in np.arange(len(Rows)):
    testTables_RMSE[i] = RMSEdf(RHcanopy['canopy_h_9'],RHcanopy[Rows[i]])
    testTables_rRMSE[i] = testTables_RMSE[i]/(np.nanmax(RHcanopy[Rows[i]]) - np.nanmin(RHcanopy[Rows[i]]))
testTables_RMSE_df = pd.DataFrame(testTables_RMSE,columns=['RMSE'],index=Rows) 
testTables_rRMSE_df = pd.DataFrame(testTables_rRMSE,columns=['rRMSE'],index=Rows)        

testTables_eval = pd.concat((round(testTables_RMSE_df,2),round(testTables_rRMSE_df,2),round(testTables_corr_pearson,2)),axis=1)
testTables_eval.to_csv(result_path+'ModelEval_ALS_and_ICESat2'+beam +'.csv')
print(testTables_eval)