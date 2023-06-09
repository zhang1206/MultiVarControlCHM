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

# path = 'C:/WorkFolder_Tianqi/CanopyHeightModel/studyRegion2_new/' 
path = 'D:/Workfolder_Zhang/Data/CanopyHeightModel/studyRegion2_new/'
# use your path
os.chdir(path)

result_path = path+'results_studyRegion3_allseason/'
if osp.exists(result_path) == False:
    os.makedirs(result_path)   
    
dataDir = 'Data_2019_2020/'
ICESatDr = 'ICESat_allseason/'

#%%
DSM = rasterio.open(path + dataDir +'elevation.tif') # 
DSMm = DSM.read(1)

TreeFraction = io.imread(path + dataDir +'TreeFraction_30.tif')
landcover = io.imread(path + dataDir +'Landcover_30.tif')
CHMrefer = io.imread(path + dataDir +'previousData/ALS_CHM_30m_RH_merged.tif')[:,:,3] # RH95

slope = io.imread(path + dataDir +'slope.tif') # ArcticDEM 
slope = np.where(slope<-100,np.nan,slope)
slopeR = slope*(math.pi/180)

roughness = np.loadtxt(path + dataDir +'roughness_DSM_ws5.tif')
NLCD = io.imread(path + dataDir +'NLCD_2016_proj.tif')

#%% Sentinel-1
Sentineldir = 'Sentinel/'

S1_VH_median = io.imread(path + dataDir + Sentineldir + 'S1_VH_median.tif.tif')
S1_VV_median = io.imread(path + dataDir + Sentineldir +'S1_VV_median.tif.tif')

VH_normalized = preProcessing.normalize(S1_VH_median)
VV_normalized = preProcessing.normalize(S1_VV_median)

# S1-deived indices"
VVHH = VV_normalized/VH_normalized
VVH =  VV_normalized * (3 ** VH_normalized) #  for compared study, change to 7

# Sentinel-2
S2 = io.imread(path + dataDir + Sentineldir + 'S2_spectra.tif.tif')

# spectra indices
S2_NDVI_median = io.imread(path + dataDir + Sentineldir + 'S2_NDVI_median.tif.tif')
S2_NDVI_max = io.imread(path + dataDir + Sentineldir + 'S2_NDVI_max2.tif.tif')
S2_NDVI_mean = io.imread(path + dataDir + Sentineldir + 'S2_NDVI_mean2.tif.tif')
S2_NDVI_std = io.imread(path + dataDir + Sentineldir + 'S2_NDVI_std2.tif.tif')

S2_NDVIredEdges1 = io.imread(path + dataDir + Sentineldir + 'S2_NDVIredEdge1.tif.tif')
S2_NDVIredEdges2 = io.imread(path + dataDir + Sentineldir + 'S2_NDVIredEdge2.tif.tif')
S2_NDVIredEdges3 = io.imread(path + dataDir + Sentineldir + 'S2_NDVIredEdge3.tif.tif')
S2_NDVIredEdges4 = io.imread(path + dataDir + Sentineldir + 'S2_NDVIredEdge4.tif.tif')

S2_EVI_median = io.imread(path + dataDir + Sentineldir +'S2_EVI.tif.tif')
S2_MSAVI_median = io.imread(path + dataDir + Sentineldir +'S2_msavi2.tif.tif')
S2_NDWI1_median = io.imread(path + dataDir + Sentineldir +'S2_NDWI1.tif.tif')
S2_NDWI2_median = io.imread(path + dataDir + Sentineldir +'S2_NDWI2.tif.tif')

S2_NDVIstdS =io.imread(path + dataDir + Sentineldir +'S2_NDVIstdS.tif')
S2_NDVIstdS3 = io.imread(path + dataDir + Sentineldir + 'S2_NDVIstdS3.tif')

#%% texture indices
S2_texture = io.imread(path + dataDir + Sentineldir +'S2_NDVI_glcm.tif.tif')
S2_contrast = S2_texture[:,:,0]
S2_corr = S2_texture[:,:,1]
S2_var = S2_texture[:,:,2]
S2_ent = S2_texture[:,:,3]
S2_diss = S2_texture[:,:,4]

#%% load ICESat-2 data ----------------------- 
beam = 'weak'
ICESatPts = gpd.read_file(path+dataDir+ICESatDr+'ICESat_'+beam+'_proj2.shp')
ICESatPts = ICESatPts.drop(['geometry'], axis = 1)

ICESatPtsXcoord = ICESatPts['xcoord'].tolist()
ICESatPtsYcoord = ICESatPts['ycoord'].tolist()

# retrieve the corresponding row and col index of ICESat pts
ICESatRow,ICESatCol = DSM.index(ICESatPtsXcoord,ICESatPtsYcoord)
ICESatPts['RowIdx'] = ICESatRow
ICESatPts['ColumnIdx'] = ICESatCol

#%-- original RHcanopy ---
temp = ICESatPts.copy()
RHcanopyO = temp[['xcoord','ycoord','SNR', 'canopy_h_9', 'canopy_h_m','canopy_h_u','RowIdx', 'ColumnIdx']]
RHcanopyO = RHcanopyO.rename(columns={'canopy_h_9':'RH98','canopy_h_m':'RH100'})

flag_validCH = (RHcanopyO['RH98']<40)&(RHcanopyO['RH98']>3)
RHcanopyO = RHcanopyO[flag_validCH]

R_RH, C_RH = RHcanopyO["RowIdx"],RHcanopyO["ColumnIdx"]
RHcanopyO['CHMrefer'] = CHMrefer[R_RH, C_RH]
RHcanopyO['NDVIstdS3'] = S2_NDVIstdS3[R_RH, C_RH]
flag_NDVIstd = RHcanopyO['NDVIstdS3']<=0.03
dataO = RHcanopyO[flag_NDVIstd].copy().dropna()

#%% load ICESat-2 data ----------------------- (quality control in our study) -----------------------\\
#% RHcanopy by our method
# urban flag: remove urban area
ICESatPts = ICESatPts[(ICESatPts['urban_flag']==0) &  (ICESatPts['canopy_h_9']<1e6)]

RHcanopy = ICESatPts[['xcoord','ycoord','SNR', 'canopy_h_9', 'canopy_h_m','canopy_h_u','RowIdx', 'ColumnIdx']]
RHcanopy = RHcanopy.rename(columns={'canopy_h_9':'RH98','canopy_h_m':'RH100'})

flag_validCH = (RHcanopy['RH98']<40)&(RHcanopy['RH98']>3)&(RHcanopy['canopy_h_u']<1e6) & (RHcanopy['RH98']<RHcanopy['RH100'])
RHcanopy = RHcanopy[flag_validCH]

# compile all features
R_RH, C_RH = RHcanopy["RowIdx"],RHcanopy["ColumnIdx"]
RHcanopy['slope'] = slope[R_RH, C_RH]
RHcanopy['elevation'] = DSMm[R_RH, C_RH]
RHcanopy['roughness'] = roughness[R_RH, C_RH]
RHcanopy['landcover'] = landcover[R_RH, C_RH]
RHcanopy['treeFraction'] = TreeFraction[R_RH, C_RH]
RHcanopy['CHU'] = RHcanopy['canopy_h_u']/RHcanopy['RH100']
RHcanopy['CHMrefer'] = CHMrefer[R_RH, C_RH]
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

#%% --- compute the cook's distance by fitting a regression model ---
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
print("Shape of data: ", data.shape)

#%%
features =  list(Sample.columns[4:])
features += ['SNR']
y = data['RH98']
x = data[features].drop(['CHMrefer'],axis=1)
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
# dataRG = data.copy()[data['cooks']<thres]
dataOr = data.copy()
dataRG = dataOr.copy()[dataOr['cooks']<thres]
print('# of Samples after cooks:',dataRG.shape)

#%%    
## initial check
RHcanopy_validCHM = dataO[['RH98','CHMrefer']]
RHcanopy_validCHM_RG = dataRG.copy()[['RH98','CHMrefer']]
RHcanopy_validCHM_copy = RHcanopy_validCHM_RG.copy()
def ModelEval(x,y):  
    R = stats.pearsonr(x,y)
    reg = LinearRegression().fit(x[:,np.newaxis], y) 
    # reg = RANSACRegressor().fit(x[:,np.newaxis], y) 
    r2 =  round(reg.score(x[:,np.newaxis], y),3)
    pred = reg.predict(x[:,np.newaxis])
    RMSE = round(np.sqrt(metrics.mean_squared_error(x,y)),2)
    # MAE = round(metrics.mean_absolute_error(x,y),2)
    rRMSE = RMSE/(np.nanmax(y)-np.nanmin(y))
    # rMAE = MAE/(np.nanmax(y)-np.nanmin(y))
    return pred,RMSE,rRMSE,R,r2

pred_RH,RMSE_RH,rRMSE_RH,R_RH,r2_RH = ModelEval(RHcanopy_validCHM['RH98'],RHcanopy_validCHM['CHMrefer'])
pred_RHf,RMSE_RHf,rRMSE_RHf,R_RHf,r2_RHf = ModelEval(RHcanopy_validCHM_copy['RH98'],RHcanopy_validCHM_copy['CHMrefer'])

textstr2 = '\n'.join((
    # r'$r^2=%.2f$' % (r2_RHf, ),
    r'$R={:.2f}$'.format(R_RHf[0]),
    r'$RMSE=%.2f$m' % (RMSE_RHf, ),
    r'$rRMSE=%.2f$' % (rRMSE_RHf, ),
    r'$Count = %.f$'%(len(RHcanopy_validCHM_copy),)))

xmin,xmax,ymin,ymax=0,32, 0,32

fig, ax = plt.subplots(figsize=(4,3.5))
fig.subplots_adjust(top=0.9,
bottom=0.158,
left=0.15,
right=0.975,
hspace=0.2,
wspace=0.2)
ax.plot([0,45],[0,45],'--k', linewidth=1.5)
ax.scatter(RHcanopy_validCHM['RH98'],RHcanopy_validCHM['CHMrefer'], s=12, alpha=0.65, c='grey',marker='.')
ax.plot(RHcanopy_validCHM['RH98'], pred_RH,'-b',linewidth=1.5)
ax.scatter(RHcanopy_validCHM_copy['RH98'],RHcanopy_validCHM_copy['CHMrefer'], s=12, alpha=0.85, c='salmon', marker='.')
ax.plot(RHcanopy_validCHM_copy['RH98'], pred_RHf,c = 'red',linestyle='-',linewidth=1.5)
ax.set_xlim([2,43])
ax.set_ylim([2,35])
ax.set_xticks([2,10,20,30,40])
ax.set_yticks([2,10,20,30,])

props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text(0.56, 0.32, textstr2, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.tight_layout()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
plt.savefig(result_path+'ScatterPlot_LiDAR_'+beam+'_criticalDi.png',dpi = 500)

#%% globally refined samples
for q in [0.7, 0.8, 0.9]:
    dataRG = data.copy()[data['cooks']<np.quantile(data['cooks'],q)]
    
    ## initial check
    RHcanopy_validCHM = dataO[['RH98','CHMrefer']]
    RHcanopy_validCHM_RG = dataRG.copy()[['RH98','CHMrefer']]
    RHcanopy_validCHM_copy = RHcanopy_validCHM_RG.copy()
    def ModelEval(x,y):  
        R = stats.pearsonr(x,y)
        reg = LinearRegression().fit(x[:,np.newaxis], y) 
        # reg = RANSACRegressor().fit(x[:,np.newaxis], y) 
        r2 =  round(reg.score(x[:,np.newaxis], y),3)
        pred = reg.predict(x[:,np.newaxis])
        RMSE = round(np.sqrt(metrics.mean_squared_error(x,y)),2)
        # MAE = round(metrics.mean_absolute_error(x,y),2)
        rRMSE = RMSE/(np.nanmax(y)-np.nanmin(y))
        # rMAE = MAE/(np.nanmax(y)-np.nanmin(y))
        return pred,RMSE,rRMSE,R,r2
    
    pred_RH,RMSE_RH,rRMSE_RH,R_RH,r2_RH = ModelEval(RHcanopy_validCHM['RH98'],RHcanopy_validCHM['CHMrefer'])
    pred_RHf,RMSE_RHf,rRMSE_RHf,R_RHf,r2_RHf = ModelEval(RHcanopy_validCHM_copy['RH98'],RHcanopy_validCHM_copy['CHMrefer'])
    
    textstr2 = '\n'.join((
        # r'$r^2=%.2f$' % (r2_RHf, ),
        r'$R={:.2f}$'.format(R_RHf[0]),
        r'$RMSE=%.2f$m' % (RMSE_RHf, ),
        r'$rRMSE=%.2f$' % (rRMSE_RHf, ),
        r'$Count = %.f$'%(len(RHcanopy_validCHM_copy),)))
    
    xmin,xmax,ymin,ymax=0,32, 0,32
    
    fig, ax = plt.subplots(figsize=(4,3.5))
    fig.subplots_adjust(top=0.9,
    bottom=0.158,
    left=0.15,
    right=0.975,
    hspace=0.2,
    wspace=0.2)
    ax.plot([0,45],[0,45],'--k', linewidth=1.5)
    ax.scatter(RHcanopy_validCHM['RH98'],RHcanopy_validCHM['CHMrefer'], s=12, alpha=0.65, c='grey',marker='.')
    ax.plot(RHcanopy_validCHM['RH98'], pred_RH,'-b',linewidth=1.5)
    ax.scatter(RHcanopy_validCHM_copy['RH98'],RHcanopy_validCHM_copy['CHMrefer'], s=12, alpha=0.85, c='salmon', marker='.')
    ax.plot(RHcanopy_validCHM_copy['RH98'], pred_RHf,c = 'red',linestyle='-',linewidth=1.5)
    ax.set_xlim([2,43])
    ax.set_ylim([2,35])
    ax.set_xticks([2,10,20,30,40])
    ax.set_yticks([2,10,20,30,])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.56, 0.32, textstr2, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()
    plt.savefig(result_path+'ScatterPlot_LiDAR_'+beam+'_'+str(q)+'_new.png',dpi = 500)