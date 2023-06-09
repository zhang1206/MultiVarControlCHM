#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:39:29 2021

@author: zhangtianqi
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
from skimage import io
import os
import os.path as osp
import matplotlib.pyplot as plt
import rasterio
# import pandas as pd
import geopandas as gpd
import math
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy import stats
import statsmodels.api as sm

path = 'D:/Workfolder_Zhang/Data/CanopyHeightModel/studyRegion5/'
# use your path
os.chdir(path)

result_path = path+'results_allseason/'
if osp.exists(result_path) == False:
    os.makedirs(result_path)   
    
dataDir = 'satellite_imagery/'
ICESatDr = 'ICESat/allseason/tracks/'

#%%
DSM = rasterio.open(path + dataDir +'elevation.tif') # 
DSMm = DSM.read(1)

TreeFraction = io.imread(path + dataDir +'TreeFraction_30.tif')
landcover = io.imread(path + dataDir +'Landcover_30.tif')
CHMrefer = io.imread(path + dataDir +'CHM_RH_2.tif')[:,:,4] # RH98
CHMrefer = np.where(CHMrefer<3,np.nan,CHMrefer)

slope = io.imread(path + dataDir +'slope.tif') # ArcticDEM 
slope = np.where(slope<-100,np.nan,slope)
slopeR = slope*(math.pi/180)

#%%
# from scipy.ndimage import generic_filter
# elevMax= generic_filter(DSMm, np.max, size=5)
# roughness = elevMax - DSMm
# CCRaster = rasterio.open(path + dataDir +'roughness_DSM_ws5.tif','w',driver='Gtiff',
#                           width=DSM.width, 
#                           height = DSM.height, 
#                           count=1, crs=DSM.crs, 
#                           transform=DSM.transform, 
#                           dtype='float64')
# CCRaster.write(roughness,1)
# CCRaster.close()

roughness = io.imread(path + dataDir +'roughness_DSM_ws5.tif')
NLCD = io.imread(path + dataDir +'NLCD_2016_proj.tif')

#%% Sentinel-1
def normalize(img):
    img_min = np.nanmin(img)
    img_max = np.nanmax(img)
    result = (img - img_min)/(img_max - img_min)
    return result

S1_VH_median = io.imread(path + dataDir + 'S1_VH_median.VH_median.tif')
S1_VV_median = io.imread(path + dataDir + 'S1_VV_median.VV_median.tif')

VH_normalized = normalize(S1_VH_median)
VV_normalized = normalize(S1_VV_median)

# S1-deived indices"
VVHH = VV_normalized/VH_normalized
VVH =  VV_normalized * (3 ** VH_normalized) #  for compared study, change to 7

# Sentinel-2
S2 = io.imread(path + dataDir + 'S2_spectra.tif')

# spectra indices
S2_NDVI_median = io.imread(path + dataDir + 'S2_NDVI_median.tif')
S2_NDVI_max = io.imread(path + dataDir + 'S2_NDVI_max2.tif')
S2_NDVI_mean = io.imread(path + dataDir + 'S2_NDVI_mean2.tif')
S2_NDVI_std = io.imread(path + dataDir +'S2_NDVI_std2.tif')

S2_NDVIredEdges1 = io.imread(path + dataDir + 'S2_NDVIredEdge1.tif')
S2_NDVIredEdges2 = io.imread(path + dataDir +  'S2_NDVIredEdge2.tif')
S2_NDVIredEdges3 = io.imread(path + dataDir + 'S2_NDVIredEdge3.tif')
S2_NDVIredEdges4 = io.imread(path + dataDir + 'S2_NDVIredEdge4.tif')

S2_EVI_median = io.imread(path + dataDir + 'S2_EVI.tif')
S2_MSAVI_median = io.imread(path + dataDir + 'S2_msavi2.tif')
S2_NDWI1_median = io.imread(path + dataDir + 'S2_NDWI1.tif')
S2_NDWI2_median = io.imread(path + dataDir + 'S2_NDWI2.tif')

#%% NDVIstd
# S2_NDVIstdS = generic_filter(S2_NDVI_median, np.std, size=7)
# S2_NDVIstdS3 = generic_filter(S2_NDVI_median, np.std, size=3)

# CCRaster = rasterio.open(path + dataDir +'S2_NDVIstdS.tif','w',driver='Gtiff',
#                           width=DSM.width, 
#                           height = DSM.height, 
#                           count=1, crs=DSM.crs, 
#                           transform=DSM.transform, 
#                           dtype='float64')
# CCRaster.write(S2_NDVIstdS,1)
# CCRaster.close()

# CCRaster = rasterio.open(path + dataDir +'S2_NDVIstdS3.tif','w',driver='Gtiff',
#                           width=DSM.width, 
#                           height = DSM.height, 
#                           count=1, crs=DSM.crs, 
#                           transform=DSM.transform, 
#                           dtype='float64')
# CCRaster.write(S2_NDVIstdS3,1)
# CCRaster.close()

S2_NDVIstdS =io.imread(path + dataDir + 'S2_NDVIstdS.tif')
S2_NDVIstdS3 = io.imread(path + dataDir + 'S2_NDVIstdS3.tif')

#%% texture indices
S2_texture = io.imread(path + dataDir +'S2_NDVI_glcm.tif')
S2_contrast = S2_texture[:,:,0]
S2_corr = S2_texture[:,:,1]
S2_var = S2_texture[:,:,2]
S2_ent = S2_texture[:,:,3]
S2_diss = S2_texture[:,:,4]

#%% -----------------compared study----------------------------------
beam = 'strong'
ICESatPts = gpd.read_file(path+ICESatDr+'ICESat_'+beam+'_proj2.shp')
ICESatPts = ICESatPts.drop(['geometry'], axis = 1)

ICESatPtsXcoord = ICESatPts['xcoord'].tolist()
ICESatPtsYcoord = ICESatPts['ycoord'].tolist()

# retrieve the corresponding row and col index of ICESat pts
ICESatRow,ICESatCol = DSM.index(ICESatPtsXcoord,ICESatPtsYcoord)
ICESatPts['RowIdx'] = ICESatRow
ICESatPts['ColumnIdx'] = ICESatCol

RHcanopy = ICESatPts[['xcoord','ycoord','SNR', 'canopy_h_9', 'canopy_h_m','canopy_h_u','RowIdx', 'ColumnIdx']]
RHcanopy = RHcanopy.rename(columns={'canopy_h_9':'RH98','canopy_h_m':'RH100'})

flag_validCH = (RHcanopy['RH100']<15)&(RHcanopy['RH100']>3)
RHcanopy = RHcanopy[flag_validCH]

# compile all features
R_RH, C_RH = RHcanopy["RowIdx"],RHcanopy["ColumnIdx"]
RHcanopy['slope'] = slope[R_RH, C_RH]
RHcanopy['elevation'] = DSMm[R_RH, C_RH]
RHcanopy['roughness'] = roughness[R_RH, C_RH]
RHcanopy['landcover'] = landcover[R_RH, C_RH]
RHcanopy['treeFraction'] = TreeFraction[R_RH, C_RH]
RHcanopy['CHU'] = RHcanopy['canopy_h_u']/RHcanopy['RH100']
# RHcanopy['CHMrefer'] = CHMrefer[R_RH, C_RH]
RHcanopy['VVH'] = VVH[R_RH, C_RH]
RHcanopy['VVHH'] = VVHH[R_RH, C_RH]
RHcanopy['VH'] = VH_normalized[R_RH, C_RH]
RHcanopy['VV'] = VV_normalized[R_RH, C_RH]

# spectra derived features
RHcanopy['NDVImed'] = S2_NDVI_median[R_RH, C_RH]
RHcanopy['NDVImax'] = S2_NDVI_max[R_RH, C_RH]
RHcanopy['NDVImean'] = S2_NDVI_mean[R_RH, C_RH]
RHcanopy['NDVIstd'] = S2_NDVI_std[R_RH, C_RH]
RHcanopy['NDVIstdS'] = S2_NDVIstdS[R_RH, C_RH]
RHcanopy['NDVIstdS3'] = S2_NDVIstdS3[R_RH, C_RH]

RHcanopy['NDWI1med'] = S2_NDWI1_median[R_RH, C_RH]
RHcanopy['NDWI2med'] = S2_NDWI2_median[R_RH, C_RH]
RHcanopy['EVImed'] = S2_EVI_median[R_RH, C_RH]
RHcanopy['MSAVImed'] = S2_MSAVI_median[R_RH, C_RH]

# texture indices: 'NDVI_contrast', 'NDVI_corr', 'NDVI_var','NDVI_ent', 'NDVI_diss' # ws = 7
RHcanopy['NDVIcontrast'] = S2_texture[:,:,0][R_RH, C_RH]
RHcanopy['NDVIcorr'] = S2_texture[:,:,1][R_RH, C_RH]
RHcanopy['NDVIvar'] = S2_texture[:,:,2][R_RH, C_RH]
RHcanopy['NDVIent'] = S2_texture[:,:,3][R_RH, C_RH]
RHcanopy['NDVIdiss'] = S2_texture[:,:,4][R_RH, C_RH]

# spectra
bandList = ['blue','green','red','redEdge1','redEdge2','redEdge3','NIR',
            'redEdge4','waterVapor','SWIR1','SWIR2']

for i in np.arange(S2.shape[2]):
    RHcanopy[bandList[i]] = S2[:,:,i][R_RH, C_RH]
    
# S2 special
RHcanopy['NDVIredEdge1'] = S2_NDVIredEdges1[R_RH, C_RH]
RHcanopy['NDVIredEdge2'] = S2_NDVIredEdges2[R_RH, C_RH]
RHcanopy['NDVIredEdge3'] = S2_NDVIredEdges3[R_RH, C_RH]
RHcanopy['NDVIredEdge4'] = S2_NDVIredEdges4[R_RH, C_RH]

RHcanopy_validICESat = RHcanopy.copy()
print(RHcanopy_validICESat.shape[0])

#%% ---- remove collinear variables -----
RHcanopy = RHcanopy_validICESat.copy()
flag = (RHcanopy['treeFraction']>0) & (RHcanopy['NDVIstdS3']<=0.03)
CHMd_S2_test = RHcanopy[flag]
CHMd_S2_nonNaN = CHMd_S2_test.dropna(axis=0)

# CHMd_nonNaN_idx =  RHcanopy[['RowIdx','ColumnIdx']]
# CHMd_nonNaN_idx.to_csv(result_path+'ALS_ICESat_intersectingPts.csv')

CHMd_S2_nonNaN = CHMd_S2_nonNaN.drop(['RH100','NDWI2med','MSAVImed','canopy_h_u',
                                      'redEdge2','redEdge3','redEdge4','VV','VH',
                                      'NIR','NDVIdiss','blue',
                                      'red','green','waterVapor',
                                      'SWIR1','NDVIstdS3'],
                                      axis=1)
print(CHMd_S2_nonNaN.shape)
Sample = CHMd_S2_nonNaN.copy()

#%%
data =  CHMd_S2_nonNaN.copy()
#-- check duplication --
df = data[['RowIdx','ColumnIdx',]]
print(sum(df[['RowIdx','ColumnIdx']].duplicated()))

# merge duplication by their minimum CHU
import pandas as pd
data_r = data.groupby(['RowIdx','ColumnIdx'])['CHU'].agg('min')
data = pd.merge(data, data_r, how="inner", on=['RowIdx','ColumnIdx','CHU'])
print("Shape of data: ", data.shape)

#%%----- construct the features for refined samples -----
#--- constructing test data with CHMrefer for validation ---
CHMrefer = io.imread(path + dataDir +'CHM_RH_2.tif')[:,:,4]
Flag_lc = (NLCD < 40)
Flag_validALS = CHMrefer<3
CHMrefer = np.where(Flag_lc|Flag_validALS, np.nan,CHMrefer)

Unknown_RowIdx, Unknown_ColumnIdx = np.where(~np.isnan(CHMrefer))
Unknown_xcoord, Unknown_ycoord = DSM.transform*(Unknown_ColumnIdx,Unknown_RowIdx)

########---- S2_AUX ----
FeaNameList = ['slope', 'elevation', 'roughness', 'landcover',
       'treeFraction', 'VVH', 'VVHH', 'NDVImed', 'NDVImax', 'NDVImean',
       'NDVIstd', 'NDVIstdS', 'NDWI1med', 'EVImed', 'NDVIcontrast', 'NDVIcorr',
       'NDVIvar', 'NDVIent', 'redEdge1', 'SWIR2', 
       'NDVIredEdge1','NDVIredEdge2', 'NDVIredEdge3', 'NDVIredEdge4']
FeaList = [slope,DSMm, roughness,landcover, TreeFraction,VVH,VVHH,
           S2_NDVI_median,S2_NDVI_max,S2_NDVI_mean,
           S2_NDVI_std, S2_NDVIstdS,S2_NDWI1_median,S2_EVI_median,
            S2_contrast,S2_corr,S2_var,S2_ent,S2[:,:,3],S2[:,:,10],
            S2_NDVIredEdges1,S2_NDVIredEdges2,S2_NDVIredEdges3,S2_NDVIredEdges4]

Unknown = pd.DataFrame()
Unknown['RowIdx'] =  Unknown_RowIdx
Unknown['ColumnIdx'] = Unknown_ColumnIdx

for i in np.arange(len(FeaList)):
    # print(i)
    Unknown[FeaNameList[i]] = FeaList[i][Unknown_RowIdx,Unknown_ColumnIdx]

Unknown_noNaN = Unknown.dropna()
Unknown_noNaNS = Unknown_noNaN
Unknown_noNaNSS = Unknown_noNaNS
Unknown_noNaN_features = Unknown_noNaNSS.to_numpy()

#%% CHM regression 
Rgroup = []
RMSEgroup = []
rRMSEgroup = []
regressor = []

#--- RF -----
Model = 'RF'
base_model = RandomForestRegressor(n_estimators = 200,max_features = 10,random_state=42)

#--- GBM --
# from sklearn.ensemble import GradientBoostingRegressor
# Model = 'GBM'
# base_model = GradientBoostingRegressor(n_estimators=200,max_features = 10,max_depth=6,random_state=42)

#-- SVM ---
# Model = 'SVM'
# from sklearn import svm
# from numpy.random import seed
# seed(1)
# base_model = svm.SVR(kernel = 'linear')

# Model = 'SVM_eps3_'
# from sklearn import svm
# from numpy.random import seed
# seed(1)
# base_model = svm.SVR(kernel = 'linear', epsilon = 3)

#--- ANN ---
# from numpy.random import seed
# seed(1)
# import tensorflow as tf
# tf.random.set_seed(1)
# Model = 'ANN'
# from keras.layers import Dense
# from keras.models import Sequential

features =  list(Sample.columns[4:])
features += ['SNR']

##%% training samples
### --- ICESat data ---- #### 
sampleStr = 'ICESat'
y = data['RH98']
x = data[features].drop(['CHU','SNR'],axis=1)
print(x.shape[0])

### ---- ALS data ---- ###
# sampleStr = 'ALS'
# y = data['CHMrefer']
# x = data[features].drop(['CHU','SNR','CHMrefer'],axis=1)

#----- 
# GLM 
# glm_gaussian = sm.GLM(y,x, family=sm.families.Gaussian())
# base_model = glm_gaussian.fit()
# Unknown_pred_rf = base_model.predict(Unknown_noNaN_features)

# -- RF, GBM, SVM  ---
base_model.fit(x,y) 
Unknown_pred_rf = base_model.predict(Unknown_noNaN_features)

#-- ANN ---
# base_model = Sequential()
# base_model.add(Dense(11, activation = 'relu', input_dim = x.shape[1]))
# base_model.add(Dense(units = 11, activation = 'relu'))
# base_model.add(Dense(units = 11, activation = 'relu'))
# base_model.add(Dense(units = 11, activation = 'relu'))
# base_model.add(Dense(1))
# base_model.compile(loss='mean_squared_error', optimizer='adam')

# base_model.fit(x,y,batch_size = 32, epochs = 300)
# Unknown_pred_rf = base_model.predict(Unknown_noNaN_features)
# Unknown_pred_rf = Unknown_pred_rf[:,0]

# model evaluation for CHM prediction
# entire region
yTrue = CHMrefer[Unknown_noNaN['RowIdx'],Unknown_noNaN['ColumnIdx']]

R_pearson = stats.pearsonr(Unknown_pred_rf,yTrue)
reg = LinearRegression().fit(Unknown_pred_rf[:,np.newaxis], yTrue)
r2 =  reg.score(Unknown_pred_rf[:,np.newaxis], yTrue)
ypred = reg.predict(Unknown_pred_rf[:,np.newaxis])

errors_rf = abs(yTrue - ypred)
RMSE_rf = np.sqrt(metrics.mean_squared_error(yTrue[:], ypred[:]))
rRMSE_rf = RMSE_rf/(np.nanmax(yTrue)-np.nanmin(yTrue))

Rgroup.append(R_pearson[0])
RMSEgroup.append(RMSE_rf)
rRMSEgroup.append(rRMSE_rf)
regressor.append(Model)

metrics = pd.DataFrame(data = {'regressor':Model,'R':Rgroup,
                                'RMSE':RMSEgroup,'rRMSE':rRMSEgroup,
                                'count':x.shape[0]})
metrics.to_csv(result_path+'regressionMetrics_original_'+Model+'_'+sampleStr+'_'+beam+'.csv')

#%% plot the optimal result
textstr = '\n'.join((
    # r'$r^2=%.2f$' % (r2, ),
    r'$R={:.2f}$'.format(R_pearson[0]),
    r'$RMSE=%.2f$m' % (RMSE_rf, ),
    r'$rRMSE=%.2f$' % (rRMSE_rf, ),
    r'$Count = %.f$'%(x.shape[0])))

import matplotlib
# plt.figure()
fig, ax = plt.subplots(figsize=(4,3))
ax.set_facecolor('xkcd:white')
cmap = matplotlib.cm.get_cmap('viridis')
cmap.set_under('w')
h=ax.hist2d(Unknown_pred_rf, yTrue, bins=150,cmap = cmap,vmin=1)
ax.set_ylim([3,15])
ax.set_xlim([3,15])
cbar = fig.colorbar(h[3], ax=ax)
tick_font_size = 12
cbar.ax.tick_params(labelsize=tick_font_size)

ax.plot(Unknown_pred_rf,ypred,'-r',linewidth=1.5)
ax.plot([3,15],[3,15],'--k', linewidth=1.5)

# ax.set_ylabel('ALS Observed (m)',fontsize=13)
# ax.set_xlabel('ICESat-2 Predicted (m)',fontsize=13)
ax.set_xticks([5,10,15])
ax.set_yticks([5,10,15])
# plt.suptitle('Controlled (ours)',fontsize=14,y=0.94,x=0.47)
# ax.set_title('Controlled (ours)',fontsize=12)

props = dict(facecolor='white', alpha=0.8)
t = ax.text(0.03, 0.95, textstr, transform=ax.transAxes, fontsize=12,verticalalignment='top', bbox=props)
t.set_bbox(dict(facecolor='white', alpha=0.8,edgecolor='w'))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(False)
plt.box('on')
plt.tight_layout()
plt.show()
plt.savefig(result_path+'CHMregression_'+sampleStr+'_original_LidarArea_'+Model+'_'+beam+'.png',dpi = 500)


