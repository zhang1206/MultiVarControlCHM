#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:39:29 2021

@author: zhangtianqi
"""

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

import sys
sys.path.append("D:\Workfolder_Zhang\myCodes\DTMextraction") 
sys.path.append("D:\Workfolder_Zhang\myCodes\DTMextraction_ArcticDEM_ICESat2_edgePixels")
import preProcessing

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

#%% load ICESat-2 data ----------------------- (quality control in our study) -----------------------
ICESatPts = gpd.read_file(path+ICESatDr+'ICESat_all_proj2.shp')
ICESatPts = ICESatPts.drop(['geometry'], axis = 1)
# print(len(ICESatPts))

ICESatPtsXcoord = ICESatPts['xcoord'].tolist()
ICESatPtsYcoord = ICESatPts['ycoord'].tolist()

# retrieve the corresponding row and col index of ICESat pts
ICESatRow,ICESatCol = DSM.index(ICESatPtsXcoord,ICESatPtsYcoord)
ICESatPts['RowIdx'] = ICESatRow
ICESatPts['ColumnIdx'] = ICESatCol

#%-- original RHcanopy ---
# RHcanopyO = ICESatPts[['xcoord','ycoord','SNR', 'canopy_h_9', 'canopy_h_m','canopy_h_u','RowIdx', 'ColumnIdx']]
# RHcanopyO = RHcanopyO.rename(columns={'canopy_h_9':'RH98','canopy_h_m':'RH100'})

# flag_validCH = (RHcanopyO['RH98']<40)&(RHcanopyO['RH98']>3)
# RHcanopyO = RHcanopyO[flag_validCH]

#% RHcanopy by our method
# urban flag: remove urban area
Flag_nonUrban = ICESatPts['urban_flag']==0
Flag_validCH = ICESatPts['canopy_h_9']<1e6
Flag_valid = Flag_nonUrban & Flag_validCH 
ICESatPts = ICESatPts[Flag_valid]
print(sum(Flag_valid))

RHcanopy = ICESatPts[['xcoord','ycoord','SNR', 'canopy_h_9', 'canopy_h_m','canopy_h_u','RowIdx', 'ColumnIdx']]
RHcanopy = RHcanopy.rename(columns={'canopy_h_9':'RH98','canopy_h_m':'RH100'})

flag_validCH = (RHcanopy['RH98']<15)&(RHcanopy['RH98']>3)&(RHcanopy['canopy_h_u']<1e6) & (RHcanopy['RH98']<RHcanopy['RH100'])
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

#%% ---- remove collinear variables -----
flag = (RHcanopy['treeFraction']>0) & (RHcanopy['NDVIstdS3']<=0.03)
CHMd_S2_test = RHcanopy[flag]

CHMd_S2_nonNaN = CHMd_S2_test.dropna(axis=0)
CHMd_S2_nonNaN = CHMd_S2_nonNaN.drop(['RH100','NDWI2med','MSAVImed','canopy_h_u',
                                      'redEdge2','redEdge3','redEdge4','VV','VH',
                                      'NIR','NDVIdiss','blue',
                                      'red','green','waterVapor',
                                      'SWIR1','NDVIstdS3'],
                                      axis=1)
print(CHMd_S2_nonNaN.shape)
Sample = CHMd_S2_nonNaN.copy()

#%% ------------------------------ feature selection -------------------------
# ---- multi-variate regression, outlier detection -----
# areas with both ALS and ICESat
# Assigning filtered data back to our original variable
data =  CHMd_S2_nonNaN.copy()

#-- check duplication --
df = data[['RowIdx','ColumnIdx',]]
print(sum(df[['RowIdx','ColumnIdx']].duplicated()))

# merge duplication by their minimum CHU
import pandas as pd
data_r = data.groupby(['RowIdx','ColumnIdx'])['CHU'].agg('min')
data = pd.merge(data, data_r, how="inner", on=['RowIdx','ColumnIdx','CHU'])
print("Shape of data: ", data_r.shape)

#%% --- compute the cook's distance by fitting a regression model ---
features =  list(Sample.columns[4:])
features += ['SNR']
y = data['RH98']
x = data[features]
print('features:',features)
print(x.shape)

from sklearn import preprocessing
# x = x.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x = min_max_scaler.fit_transform(x)

#%% remove outliers by Cook's Distance
# --- model fit ---
glm_gaussian = sm.GLM(y,x, family=sm.families.Gaussian())
model = glm_gaussian.fit()
# print(model.summary())

#--- cook's distance calculation ---
GLMinf = model.get_influence(observed=False)
np.set_printoptions(suppress=True)

cooks = GLMinf.cooks_distance
data['cooks'] = cooks[0]
data = data.dropna()

thres = np.quantile(cooks[0],0.75) - np.quantile(cooks[0],0.25)*1.5 + np.quantile(cooks[0],0.75)

#%%
fig, ax = plt.subplots(figsize=(6,3))
plt.rcParams.update(plt.rcParamsDefault)

#----scatter
plt.subplot(121)
plt.scatter(np.arange(0,data.shape[0]), cooks[0]**0.25,s=1,alpha = 0.5,c='grey')
plt.xlabel('Observation number',fontsize = 12)
plt.ylabel(r"$\sqrt[4]{\rm{D_{i}}}$",fontsize = 12)
plt.axhline(y= thres**0.25, color='r', 
            linestyle='--',label='Critical '+ r'$D_{i}$') #alpha = 0.05
plt.xlim([0, x.shape[0]+1])
# plt.xticks([500,1000,1500, 2000, 2500]) # all
# plt.xticks([100, 300, 500, 700]) # weak
# plt.xticks([500, 1000,1500, 2000]) # strong

#----histogram
plt.subplot(122)
plt.hist(cooks[0]**0.25,bins=150,color='grey',density=True)
plt.xlabel(r"$\sqrt[4]{\rm{D_{i}}}$",fontsize = 12)
plt.ylabel('Probability Density',fontsize = 12)
# plt.axvline(x=np.quantile(cooks[0]**0.25,0.8), color='g', 
#             linestyle='-',label='0.8 quantile') #alpha = 0.05
# plt.axvline(x=np.quantile(cooks[0]**0.25,0.9), color='b', 
#             linestyle='-',label='0.9 quantile') #alpha = 0.05
plt.axvline(x= thres**0.25, color='r', 
            linestyle='--',label='Critical '+ r'$D_{i}$') #alpha = 0.05
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig(result_path+'CookDist.png',dpi = 500)

#%%----- construct the features for refined samples -----
# original samples without refinement: data_rr
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

##%% CHM regression 
Rgroup = []
RMSEgroup = []
rRMSEgroup = []

#%% CHM regression 
# -- GLM ---
# Model = 'GLM'

#--- RF -----
Model = 'RF'
base_model = RandomForestRegressor(n_estimators = 200,max_features = 10,random_state=42)

#--- GBM --
# from sklearn.ensemble import GradientBoostingRegressor
# Model = 'GBM'
# base_model = GradientBoostingRegressor(n_estimators=200,max_features = 10,max_depth=6,random_state=42)

#-- SVM ---
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

# # CHM regression 
# #--- quality refined data ---
dataO = data.copy()
dataRG = dataO.copy()[dataO['cooks']<thres]
print('# of Samples after cooks:',dataRG.shape)

### --- ICESat data ---- #### 
sampleStr = 'ICESat'
y = dataRG['RH98']
x = dataRG[features].drop(['CHU','SNR'],axis=1)

#### ---- ALS data ---- ###
# sampleStr = 'ALS'
# y = dataRG['CHMrefer']
# x = dataRG[features].drop(['CHU','SNR','CHMrefer'],axis=1)

# GLM 
# glm_gaussian = sm.GLM(y,x, family=sm.families.Gaussian())
# base_model = glm_gaussian.fit()
# Unknown_pred_rf = base_model.predict(Unknown_noNaN_features)

# -- RF, GBM, SVM ---
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
yTrue = CHMrefer[Unknown_noNaN['RowIdx'],Unknown_noNaN['ColumnIdx']]

R_pearson = stats.pearsonr(Unknown_pred_rf,yTrue)
reg = LinearRegression().fit(Unknown_pred_rf[:,np.newaxis], yTrue)
r2 =  reg.score(Unknown_pred_rf[:,np.newaxis], yTrue)
ypred = reg.predict(Unknown_pred_rf[:,np.newaxis])

errors_rf = abs(yTrue - ypred)
RMSE_rf = np.sqrt(metrics.mean_squared_error(yTrue, ypred))
rRMSE_rf = RMSE_rf/(np.nanmax(yTrue)-np.nanmin(yTrue))

Rgroup.append(R_pearson[0])
RMSEgroup.append(RMSE_rf)
rRMSEgroup.append(rRMSE_rf)

metrics = pd.DataFrame(data = {'regressor':Model,'R':Rgroup,
                                'RMSE':RMSEgroup,'rRMSE':rRMSEgroup,
                                'count':x.shape[0]})
metrics.to_csv(result_path+'regressionMetrics_ours_'+Model+'_'+sampleStr+'.csv')

#%% feature importance
# from sklearn.inspection import permutation_importance
# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     MAE = np.mean(errors)
    
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     R_pearson = stats.pearsonr(test_labels, predictions)
    
#     print('corr = {:0.2f}.'.format(R_pearson[0]))
#     print('Model Performance')
#     print('MAE: {:0.4f} m.'.format(MAE))
#     print('Accuracy = {:0.2f}.'.format(accuracy))
    
#     return accuracy, R_pearson[0], MAE

# # base_model.fit(x, y) 
# result = permutation_importance(base_model, x, y, random_state=42,n_jobs=4)

# x_labels = x.copy().rename(columns={'RowIdx':'ycoord','ColumnIdx':'xcoord'})
# varImportance = pd.DataFrame()
# varImportance['Importance'] = list(result.importances_mean)
# varImportance['std'] = list(result.importances_std)
# varImportance['features'] = x_labels.columns
# varImportance = varImportance.set_index('features')
# varImportance = varImportance.sort_values('Importance', ascending=True)

# #%% plot the feature importance
# plt.style.use('ggplot')
# plt.rcParams.update({"lines.markeredgewidth" : 1,
#                       "errorbar.capsize" : 2,
#                       "font.family":"arial"})
# s = pd.Series(varImportance['Importance'].T)
# err = list(varImportance['std'])
# plt.figure(figsize = (3.5,5))
# s.plot(kind='barh', xerr=err)
# plt.ylabel('')
# plt.legend(loc='lower right')
# plt.axis('on')
# plt.xticks([0,0.1,0.2,0.3,0.4])
# # plt.xlabel('Feature Importance')
# plt.tight_layout()
# plt.show()
# plt.savefig(result_path+'variableImportance_'+Model+'_'+sampleStr+'.png',dpi = 500)

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
plt.savefig(result_path+'CHMregression_'+sampleStr+'_ours_lidarArea_'+Model+'.png',dpi = 500)

#%% per slope and land cover classes
# from collections import Counter
# Vallc = np.unique(NLCD[Unknown_noNaN['RowIdx'], Unknown_noNaN['ColumnIdx']])
# Countlc= Counter(NLCD[Unknown_noNaN['RowIdx'], Unknown_noNaN['ColumnIdx']])
# # print(Countlc)

# for Value in Vallc:
#     print (Value, Countlc[Value]) 

# """
# landcoverID, Count

# 20 8749 shrubs
# 30 1825 herbaceous
# # 60 123 Bare / sparse vegetation
# # 80 1569  Permanent water bodies. 
# # 90 13 Herbaceous wetland. 
# 111 89768 Closed forest, evergreen needle leaf.
# 114 177 Closed forest, deciduous broad leaf.
# 115 15155 Closed forest, mixed.
# 116 29991 Closed forest, not matching any of the other definitions.
# 121 820 Open forest, evergreen needle leaf.
# 126 18250 Open forest, not matching any of the other definitions.

# NLCDID, Count

# 20 8749 shrubs
# 30 1825 herbaceous
# 111 89768 Closed forest, evergreen needle leaf.
# 114 177 Closed forest, deciduous broad leaf.
# 115 15155 Closed forest, mixed.
# 116 29991 Closed forest, not matching any of the other definitions.
# 121 820 Open forest, evergreen needle leaf.
# 126 18250 Open forest, not matching any of the other definitions.

# """

# CHMreferLoc = pd.DataFrame()
# CHMreferLoc['RowIdx'] = Unknown_noNaN['RowIdx']
# CHMreferLoc['ColumnIdx'] = Unknown_noNaN['ColumnIdx']
# CHMreferLoc['ALS observed'] = CHMrefer[Unknown_noNaN['RowIdx'], Unknown_noNaN['ColumnIdx']]
# CHMreferLoc['landcover'] = NLCD[Unknown_noNaN['RowIdx'], Unknown_noNaN['ColumnIdx']]
# CHMreferLoc['ICESat-2 predicted'] = Unknown_pred_rf
# CHMreferLoc['slope'] = slope[Unknown_noNaN['RowIdx'], Unknown_noNaN['ColumnIdx']]

# # slope group
# G = CHMreferLoc['slope']
# G1 = G<=10
# G2 = (G>10)&(G<=15)
# G3 = (G>15)&(G<=20)
# G4 = (G>20)

# CHMreferLoc['group_slope'] = G1*1+G2*2+G3*3+G4*4

# lcIDX = (CHMreferLoc['landcover'] == 41) | (CHMreferLoc['landcover'] == 42) | (CHMreferLoc['landcover'] == 43)| (CHMreferLoc['landcover'] == 52)
# CHMreferLoc = CHMreferLoc[lcIDX]

# # canopy height over each
# # CHMreferLoc = CHMreferLoc.rename(columns={"A": "a", "B": "c"})
# # import seaborn as sns
# # dd=pd.melt(CHMreferLoc,id_vars=['landcover'],value_vars=['ALS observed', 'ICESat-2 predicted'],var_name='CanopyHeights')
# # fig, ax = plt.subplots(figsize=(7,4))
# # sns.boxplot(ax=ax,x='landcover',y='value',data=dd,hue='CanopyHeights',palette="Set2",fliersize=0.3,
# #             showfliers=False,width=0.5)
# # plt.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), borderaxespad=0.) # bbox_to_anchor (x,y) where origin is located at lower left (0,0)
# # plt.ylabel('Vegetation Height (m)',fontsize=12)
# # plt.xlabel('NLCD Land Cover',fontsize=12)
# # plt.ylim([-1,35])
# # plt.yticks([0,5,10,15,20,25,30],fontsize=11)
# # plt.xticks(np.arange(len(np.unique(CHMreferLoc['landcover']))), 
# #            ('Deciduous forest','Evergreen forest','Mixed forest','Shrub'),
# #            rotation=10,fontsize=11)

# # fig.tight_layout()
# # plt.savefig(result_path + 'canopyHeights_'+FeatStr+'.png',dpi = 500)

# #%% landcover
# predList_lc = []
# titleList_lc = ['Deciduous forest','Evergreen forest','Mixed forest','Shrub']
# for lc in np.sort(CHMreferLoc['landcover'].unique()):    
#       predList = [CHMreferLoc[CHMreferLoc['landcover'] == lc] ]
#       predList_lc.append(predList)

# Vallc = np.unique(CHMreferLoc['landcover'])
# Countlc= Counter(CHMreferLoc['landcover'])
    
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,6))
# # fig.subplots_adjust(wspace=0.25,hspace=0.5)
# # fig.subplots_adjust(top=0.875,
# # bottom=0.15,
# # left=0.11,
# # right=0.88,
# # hspace=0.5,
# # wspace=0.285)

# lc_df = []
# for i, (ax, feature,title) in enumerate(zip(axes.flatten(), predList_lc,titleList_lc)):
#     # print(feature[0].columns)
#     y_pred = feature[0]['ICESat-2 predicted']
#     y_true = feature[0]['ALS observed']
    
#     # random.seed(10)
#     # if len(y_pred)>2000:       
#     #     sampleSize = 2000
#     #     # Generate 10 unique random numbers within a range
#     #     sample_list = random.sample(range(0, y_pred.shape[0]), 2000)
#     #     y_pred = np.array(y_pred)[sample_list]
#     #     y_true = np.array(y_true)[sample_list]
        
    
#     # ymax = np.max([np.nanmax(y_pred), np.nanmax(y_true)])
#     # xmin,xmax,ymin,ymax=-1, ymax+1, -1, ymax+1
    
#     reg = LinearRegression().fit(y_pred[:,np.newaxis], y_true)
#     r2 =  reg.score(y_pred[:,np.newaxis], y_true)
#     yPred = reg.predict(y_pred[:,np.newaxis])
    
#     R_pearson = stats.pearsonr(y_pred,y_true)
    
#     errors_rf = abs(y_pred-y_true)
#     RMSE_rf = np.sqrt(metrics.mean_squared_error(y_pred,y_true))
#     rRMSE_rf = RMSE_rf/(np.nanmax(y_true)-np.nanmin(y_true))

#     textstr = '\n'.join((
#     # r'$r^2=%.2f$' % (r2, ),
#     '$R={:.2f}$'.format(R_pearson[0]),
#     r'$RMSE=%.2f$m' % (RMSE_rf, ),
#     r'$rRMSE=%.2f$' % (rRMSE_rf, ),
#     r'$ count = %.f$'%(Countlc[Vallc[i]],)))
    
#     #-- store results --
#     # lc_df = pd.DataFrame()
#     landcover_df = pd.DataFrame(np.array([round(RMSE_rf,2),round(rRMSE_rf,2),round(R_pearson[0],2),Countlc[Vallc[i]]])[np.newaxis,], 
#                                 columns=['RMSE', 'rRMSE','R','Count'],index = [title])
#     lc_df.append(landcover_df)
    
#     h=ax.hist2d(y_pred, y_true, bins=80)
#     plt.colorbar(h[3], ax=ax)
#     # ax.scatter(y_pred, y_true, s=10, alpha=0.4, c='grey',marker='.')
#     # ax.set_xlim([0,23])
#     # ax.set_ylim([2,])
    
#     ax.plot(y_pred,yPred,'-r',linewidth=1.5)
#     ax.plot([0,30],[0,30],'--k', linewidth=1.5)
    
#     ax.set_xlabel('ICESat-2 Predicted (m)',fontsize=11)
#     ax.set_ylabel('ALS Observed (m)', fontsize=11)
#     # ax.set_xticks([0,4,8,12,16])
    
#     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
#     ax.set_title(title, y = 1.05, fontsize=12)

# # plt.tight_layout()
# # plt.savefig(result_path + 'CHM_landcover_'+Model+'.png',dpi = 500)

# frame = pd.concat(lc_df, axis=0)
# frame.to_csv((result_path+'CHM_landcover_'+Model+'.csv'))

# #%% slope
# predList_lc = []
# titleList_lc = ['<10˚','10˚-15˚','15˚-20˚','>20˚']

# for g in np.sort(CHMreferLoc['group_slope'].unique()):    
#       predList = [CHMreferLoc[CHMreferLoc['group_slope'] == g] ]
#       predList_lc.append(predList)
     
# Valg = np.unique(CHMreferLoc['group_slope'])
# Countg= Counter(CHMreferLoc['group_slope'])

# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7,6))
# # fig.subplots_adjust(top=0.9,
# # bottom=0.1,
# # left=0.1,
# # right=0.95,
# # hspace=0.5,
# # wspace=0.35)

# lc_df = []
# for i, (ax, feature,title) in enumerate(zip(axes.flatten(), predList_lc,titleList_lc)):
#     # print(feature[0].columns)
#     y_pred = feature[0]['ICESat-2 predicted']
#     y_true = feature[0]['ALS observed']
    
#     # ymax = np.max([np.nanmax(y_pred), np.nanmax(y_true)])
#     # xmin,xmax,ymin,ymax=-1, ymax+1, -1, ymax+1
    
#     reg = LinearRegression().fit(y_pred[:,np.newaxis], y_true)
#     r2 =  reg.score(y_pred[:,np.newaxis], y_true)
#     yPred = reg.predict(y_pred[:,np.newaxis])
    
#     R_pearson = stats.pearsonr(y_pred,y_true)
    
#     errors_rf = abs(y_pred-y_true)
#     RMSE_rf = np.sqrt(metrics.mean_squared_error(y_pred,y_true))
#     rRMSE_rf = RMSE_rf/(np.nanmax(y_true)-np.nanmin(y_true))

#     textstr = '\n'.join((
#     # r'$r^2=%.2f$' % (r2, ),
#     '$R={:.2f}$'.format(R_pearson[0]),
#     r'$RMSE=%.2f$m' % (RMSE_rf, ),
#     r'$rRMSE=%.2f$' % (rRMSE_rf, ),
#     r'$ count = %.f$'%(Countg[Valg[i]],)))
    
#     landcover_df = pd.DataFrame(np.array([round(RMSE_rf,2),round(rRMSE_rf,2),round(R_pearson[0],2),Countlc[Vallc[i]]])[np.newaxis,], 
#                                 columns=['RMSE', 'rRMSE','R','Count'],index = [title])
#     lc_df.append(landcover_df)
    
#     h=ax.hist2d(y_pred, y_true, bins=80)
#     plt.colorbar(h[3], ax=ax)
#     # ax.scatter(y_pred, y_true, s=10, alpha=0.4, c='grey',marker='.')
#     ax.set_xlabel('ICESat-2 Predicted (m)',fontsize=11)
#     ax.set_ylabel('ALS Observed (m)', fontsize=11)
#     # ax.set_xlim([0,32])
#     # ax.set_xticks([0,5,10,15,20,25])
    
#     ax.plot(y_pred,yPred,'-r',linewidth=1.5)
#     ax.plot([0,30],[0,30],'--k', linewidth=1.5)
    
#     props = dict(boxstyle='round', facecolor='white', alpha=0.5)
#     ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
#     ax.set_title(title, y = 1.05, fontsize=12)

# # cbar_ax = fig.add_axes([0.92, 0.32, 0.02, 0.4])
# # cbar = fig.colorbar(h[3], cax=cbar_ax,ticks=([10,30,50,70]))

# # plt.tight_layout()
# # plt.savefig(result_path + 'CHM_slope_'+Model+'.png',dpi = 500)

# frame = pd.concat(lc_df, axis=0)
# frame.to_csv((result_path+'CHM_slope_'+Model+'.csv'))

# #%% predict the entire study region
# """
# use tree fraction to mask out the nonvegetation region
# """
# maskedRegion = io.imread(path + dataDir +'MaskedRegion.tif')
# maskedRegion = np.where(maskedRegion<-1000,np.nan,1)

# maskedRegion = np.where(TreeFraction<=0,np.nan,maskedRegion)
# Unknown_RowIdx_maskedRegion, Unknown_ColumnIdx_maskedRegion = np.where(~np.isnan(maskedRegion))
# Unknown_xcoord_maskedRegion, Unknown_ycoord_maskedRegion = DSM.transform*(Unknown_ColumnIdx_maskedRegion,
#                                                                           Unknown_RowIdx_maskedRegion)

# Unknown_maskedRegion = pd.DataFrame()
# Unknown_maskedRegion['RowIdx'] =  Unknown_RowIdx_maskedRegion
# Unknown_maskedRegion['ColumnIdx'] = Unknown_ColumnIdx_maskedRegion
# Unknown_maskedRegion['xcoord'] =  Unknown_xcoord_maskedRegion
# Unknown_maskedRegion['ycoord'] =  Unknown_ycoord_maskedRegion

# for i in np.arange(len(FeaList)):
#     Unknown_maskedRegion[FeaNameList[i]] = FeaList[i][Unknown_RowIdx_maskedRegion,
#                                                       Unknown_ColumnIdx_maskedRegion]

# Unknown_noNaN_maskedRegion = Unknown_maskedRegion.dropna()
# Unknown_noNaNS_maskedRegion = Unknown_noNaN_maskedRegion.drop(['RowIdx','ColumnIdx'],axis=1)
# Unknown_noNaNSS_maskedRegion = Unknown_noNaNS_maskedRegion
# Unknown_noNaN_features_maskedRegion = Unknown_noNaNSS_maskedRegion.to_numpy()
# Unknown_pred_rf_maskedRegion = base_model.predict(Unknown_noNaN_features_maskedRegion)

# RFpred_map = np.copy(maskedRegion)
# RFpred_map[Unknown_noNaN_maskedRegion['RowIdx'].astype(int).to_list(),
#            Unknown_noNaN_maskedRegion['ColumnIdx'].astype(int).to_list()] = Unknown_pred_rf_maskedRegion

# nonVegIDX = np.isnan(maskedRegion)
# RFpred_map = np.where(nonVegIDX,np.nan,RFpred_map)

# CHMrefer = io.imread(path + dataDir +'ALS_CHM_30m_RH_merged.tif')[:,:,3]
# CHMrefer = np.where(CHMrefer<0.5,np.nan,CHMrefer)

# # visualize the prediction
# import earthpy.spatial as es
# hillshade_azimuth_210 = es.hillshade(DSMm, azimuth=100)
# hillshade_azimuth_210 = np.where(nonVegIDX,np.nan,hillshade_azimuth_210)

# fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(8,8))
# plt.imshow(hillshade_azimuth_210, cmap="Greys",alpha=0.6)
# im = plt.imshow(RFpred_map,vmin=0,vmax=32,alpha=0.8)
# plt.imshow(CHMrefer,vmin=0,vmax=32,)
# axes.axis('off')
# plt.yticks([])
# plt.xticks([])
# plt.title('CHM prediction', y=1.07,fontsize=16)
# plt.grid(False) 
# plt.show()

# cbar_ax = fig.add_axes([0.78, 0.18, 0.03, 0.13])
# cbar = fig.colorbar(im, cax=cbar_ax,
#                     ticks=[0,32])

# plt.savefig(result_path+ 'CHMmap_'+ Model +'.png',dpi = 500)

# fig, axes = plt.subplots(nrows=1, ncols=1,figsize=(8,8))
# plt.imshow(hillshade_azimuth_210, cmap="Greys",alpha=0.6)
# im = plt.imshow(RFpred_map,vmin=0,vmax=32,alpha=0.8)
# # plt.imshow(CHMrefer,vmin=0,vmax=32,cmap = 'gray')
# axes.axis('off')
# plt.yticks([])
# plt.xticks([])
# plt.title('CHM prediction', y=1.07,fontsize=16)
# plt.grid(False) 
# plt.show()

# cbar_ax = fig.add_axes([0.78, 0.18, 0.03, 0.13])
# cbar = fig.colorbar(im, cax=cbar_ax,
#                     ticks=[0,32])

# plt.savefig(result_path+ 'CHMmap_'+ Model +'.png',dpi = 500)

# RFpred_Image = rasterio.open(result_path+'CHMpred_'+ Model +'.tif','w',driver='Gtiff',
#                           width=DSM.width, 
#                           height = DSM.height, 
#                           count=1, crs=DSM.crs, 
#                           transform=DSM.transform, 
#                           dtype='float64')
# RFpred_Image.write(RFpred_map,1)
# RFpred_Image.close()