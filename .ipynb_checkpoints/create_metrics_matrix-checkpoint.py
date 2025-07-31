### Create a netcdf file with a matrix of processed time series
# this works with environment npl2023b
# to run on the command line:
#
# module load conda
# conda activate npl2023b
# python create_metrics_matrix.py


# This is a script version of a python notebook


import numpy as np
import numpy.matlib
import numpy.ma as ma

import xarray as xr
#xr.set_options(enable_cftimeindex=True)
from xarray.coding.times import CFTimedeltaCoder

import time
import cftime
import netCDF4 as nc
from datetime import timedelta

import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import glob


import nc_time_axis
import cartopy.crs as ccrs
import cartopy



#--- custom files
# load custom functions for analyzing flat10

from loading_function_flat10 import load_flat10, load_grid, select_time_slice, weighted_temporal_mean 


####-------------####
# define which runs and models

outputdir= '/glade/campaign/cgd/tss/people/aswann/flat10/'

modellist_orig= ['ACCESS-ESM1-5',  
            'CESM2',    
            'GFDL-ESM4',  
            'GISS_E2.1',  
            'NorESM2-LM',
            'MPI-ESM1-2-LR',
            'CNRM-ESM2-1',
            'HadCM3LC-Bris',
            'UKESM1.2']
modellist=modellist_orig

runlist = ['flat10','flat10_zec','flat10_cdr']
# use a wildcard to capture different ways the folders and runs are named across models
runlist_wc = ['*lat10','*zec','*cdr']

varlist_load=['cVeg','cSoil','cLitter','nbp','gpp','rh','tas','pr'] #, 'gpp','fgco2', 'ra', 'rh']#, 'npp'] # not working beyond nbp for norESM
varlist_analyze=['cVeg','cSoil','cTot','cLitter','nbp','gpp','rh','tas','pr']
varlist=varlist_load
unitslist=['kgC m-2','kgC m-2','kgC m-2','kgC m-2 s-1','kgC m-2 s-1','kgC m-2 s-1','K','kg m-2 s-1']


# global and three latitude bands: trop, mid, high
latlist=['global','highlat','troplat','midlat']
troplat=30
highlat=60

## unit conversions
#unitconversions
sperday=60*60*24
speryr=60*60*24*365

PgperKg = 1e-12
# to convert kg m-2 s-1 to kg/m2/yr multiply by speryr

# initialize a dictionary to hold all of the data
data_dict={}


####---------------####

#-- load grid info
data_dict = load_grid(data_dict,modellist)


#-- load data
data_dict=load_flat10(data_dict, modellist, runlist, runlist_wc, varlist)

#-- toggle to analysis lists
modellist=modellist_orig
varlist=varlist_analyze


####---------------####
# create matrix of zonal mean time series for 

#- initialize
C_global_mat= np.empty([350,len(modellist),len(runlist),len(varlist)])
C_highlat_mat= np.empty([350,len(modellist),len(runlist),len(varlist)])
C_troplat_mat= np.empty([350,len(modellist),len(runlist),len(varlist)])
C_midlat_mat= np.empty([350,len(modellist),len(runlist),len(varlist)])

# create a time series of years for the first dimension
ts= np.arange(350)

for m in range(len(modellist)):
#for m in range(len(['GFDL-ESM4',  'GISS_E2.1',  'NorESM2-LM','MPI-ESM1-2-LR'])):
    model=modellist[m]
    print('loading model: ' +model)
    #----loop over experiments----# 
    for e in range(len(runlist)):
        run = runlist[e]
        print('processing run: ' +run)


        ds=data_dict[modellist[m] +'_' +runlist[e]]
        ds_area = data_dict[modellist[m] +'_' +'areacella']
        ds_landfrac = data_dict[modellist[m] +'_' +'landfrac']
        #area = ds_area['areacella'].expand_dims(dim={'time': ds.time.size}, axis=0)
        #landfrac=ds_landfrac['sftlf'].expand_dims(dim={'time': ds.time.size}, axis=0)
        
        
        if model=='CESM2':
            area = ds_area['areacella'].squeeze().reindex_like(ds, method='nearest',tolerance=0.05)
        else:
            area = ds_area['areacella'].reindex_like(ds, method='nearest',tolerance=0.05)
        
        landfrac=ds_landfrac['sftlf'].reindex_like(ds, method='nearest',tolerance=0.05)
        
        if landfrac.max(dim=['lat','lon'])>1: #test if landfrac is on a 0-100 or 0-1 scale
            landfrac=landfrac/100
            
        landarea=area*landfrac


        #----loop over variables----#
        for v in range(len(varlist)):
            var=varlist[v]
            #print('loading variable: ' +var)

            data_var= weighted_temporal_mean(ds, var)

            if var=='tas' or var=='pr': 
                # if this is *not* a carbon variable then we want to make an average
                C_global =(((data_var*landarea)).sum(dim=['lat','lon']))/landarea.sum(dim=['lat','lon'])
                C_highlat=(((data_var*landarea).where(ds.lat>=highlat)).sum(dim=['lat','lon']))/((landarea.where(ds.lat>=highlat)).sum(dim=['lat','lon']))
                C_troplat=(((data_var*landarea).where((ds.lat>=-troplat) & (ds.lat<=troplat))).sum(dim=['lat','lon']))/((landarea.where((ds.lat>=-troplat) & (ds.lat<=troplat))).sum(dim=['lat','lon']))
                C_midlat=(((data_var*landarea).where((ds.lat>=troplat) & (ds.lat<=highlat))).sum(dim=['lat','lon']))/((landarea.where((ds.lat>=troplat) & (ds.lat<=highlat))).sum(dim=['lat','lon']))
    
                #put into matrix 
                C_global_mat[0:len(C_global),m,e,v]= C_global
                C_highlat_mat[0:len(C_global),m,e,v]= C_highlat
                C_troplat_mat[0:len(C_global),m,e,v]= C_troplat
                C_midlat_mat[0:len(C_global),m,e,v]= C_midlat
            
            else: # it is a carbon variable and we want to make a sum
                # total carbon on land. Becuase it is in units of carbon/area (kgC/m2), multiply by area
                # our area variable is in m2
                C_global =(((data_var*landarea)).sum(dim=['lat','lon']))
                C_highlat=((data_var*landarea).where(ds.lat>=highlat)).sum(dim=['lat','lon'])
                C_troplat=((data_var*landarea).where((ds.lat>=-troplat) & (ds.lat<=troplat))).sum(dim=['lat','lon'])
                C_midlat=((data_var*landarea).where((ds.lat>=troplat) & (ds.lat<=highlat))).sum(dim=['lat','lon'])
    
                #put into matrix and convert to PgC (kgC => PgC, divide by 10^12)
                C_global_mat[0:len(C_global),m,e,v]= C_global*PgperKg
                C_highlat_mat[0:len(C_global),m,e,v]= C_highlat*PgperKg
                C_troplat_mat[0:len(C_global),m,e,v]= C_troplat*PgperKg
                C_midlat_mat[0:len(C_global),m,e,v]= C_midlat*PgperKg




####----------------####

# put the matrix into an xarray dataset
data_array_combined = np.full((len(ts), len(modellist), len(runlist), len(varlist), len(latlist)),np.nan)

data_array_combined[:,:,:,:,0]=C_global_mat
data_array_combined[:,:,:,:,1]=C_highlat_mat
data_array_combined[:,:,:,:,2]=C_troplat_mat
data_array_combined[:,:,:,:,3]=C_midlat_mat


####----------------####
# put into an xarray dataset

ds_C_global= xr.Dataset(
    {
        "data": (["time", "model", "run", "var","latrange"], data_array_combined)
    },
    coords={
        "time": ts,
        "model": modellist,
        "run": runlist,
        "var": varlist,
        "latrange": latlist
    }
)


####----------------####

# - save the matrix to a netcdf file
ds_C_global.to_netcdf("C_metrics_matrix.nc")