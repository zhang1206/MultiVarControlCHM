# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:56:48 2022

@author: zhang
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

path = 'D:/Workfolder_Zhang/Data/CanopyHeightModel/studyRegion5/'
# use your path
os.chdir(path)

import tarfile
import glob
import re
from pathlib import Path
from osgeo import gdal

folder_path = path
BASE_PATH = os.path.dirname(os.path.abspath(folder_path))
folder = Path(folder_path)

#%%
xres=30
yres=30
resample_alg = 'Average'

opt1 = gdal.WarpOptions(xRes=30, yRes=30, resampleAlg='Average')

# l = []
for f in folder.glob('./*.tif'):
    f_path = f.as_posix()
    outfn = path + 'CHM_avg_test_30m.tif'
    gdal.Warp(outfn, f_path, options=opt1)
    # ds = None
    # l.append(f_path)

#%%% -------canopy height quantile calculation ---------------------
from osgeo import gdal
DTM = rasterio.open(path+'CHM_avg_test_30m.tif') 
test_matrix = io.imread(path+'mergedCHM.tif')
test_matrix = np.where(test_matrix==0,np.nan,test_matrix)

Rowdiff = round(test_matrix.shape[0]/30)*30 - test_matrix.shape[0]
Coldiff = round(test_matrix.shape[1]/30)*30 - test_matrix.shape[1]

if (Rowdiff>0) and (Coldiff<=0):
   test = np.pad(test_matrix, ((0,Rowdiff),(0,0)), 'constant', constant_values=np.nan)
if (Rowdiff<=0) and (Coldiff>0):
   test = np.pad(test_matrix, ((0,0),(0,Coldiff)), 'constant', constant_values=np.nan)
if (Rowdiff>0) and (Coldiff>0):
   test = np.pad(test_matrix, ((0,Rowdiff),(0,Coldiff)), 'constant', constant_values=np.nan)
if (Rowdiff<=0) and (Coldiff<=0):
   test = test_matrix[:round(test_matrix.shape[0]/30)*30, :round(test_matrix.shape[1]/30)*30]

Nrow = round(test_matrix.shape[0]/30)
Ncol = round(test_matrix.shape[1]/30)
outPutMat = np.ones((Nrow,Ncol,6))*-99999
canopyCoverMat = np.ones((Nrow,Ncol))*-99999
for i in range(0, Nrow-1):
    for j in range(0, Ncol-1):
        localMatrix = test[i*30:(i+1)*30,j*30:(j+1)*30]
        loc_50, loc_75,loc_90,loc_95,loc_98,loc_100 = np.nanpercentile(localMatrix, 50), np.nanpercentile(localMatrix, 75),np.nanpercentile(localMatrix, 90),np.nanpercentile(localMatrix, 95),np.nanpercentile(localMatrix, 98),np.nanpercentile(localMatrix, 100)
        outPutMat[i,j,0] = loc_50
        outPutMat[i,j,1] = loc_75
        outPutMat[i,j,2] = loc_90
        outPutMat[i,j,3] = loc_95
        outPutMat[i,j,4] = loc_98
        outPutMat[i,j,5] = loc_100   
        canopyCoverMat[i,j] = np.sum(localMatrix>2)/9

outPutMat = np.where(outPutMat<0,np.nan,outPutMat)
canopyCoverMat = np.where(canopyCoverMat<=0,np.nan,canopyCoverMat)

#%% output product
outPutRaster = rasterio.open(path+'CHM_RH','w',driver='Gtiff',
                          width=DTM.width, 
                          height = DTM.height, 
                          count=6, crs=DTM.crs, 
                          transform=DTM.transform, 
                          dtype='float32')
outPutRaster.write(outPutMat[:,:,0],1)
outPutRaster.write(outPutMat[:,:,1],2)
outPutRaster.write(outPutMat[:,:,2],3)
outPutRaster.write(outPutMat[:,:,3],4)
outPutRaster.write(outPutMat[:,:,4],5)
outPutRaster.write(outPutMat[:,:,5],6)
outPutRaster.close()

CCRaster = rasterio.open(path+'CHM_CC','w',driver='Gtiff',
                          width= DTM.width, 
                          height =DTM.height, 
                          count=1, crs=DTM.crs, 
                          transform=DTM.transform, 
                          dtype='float32')
CCRaster.write(canopyCoverMat,1)
CCRaster.close()
