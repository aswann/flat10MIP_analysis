def load_flat10(data_dict, modellist, runlist, runlist_wc, varlist):

    '''
     this function loads all variables in varlist for all experiments in runlist and models in modellist
    
     data_dict - dictionary to hold xarrays
     modellist - list array of model names that match filenames
     runlist - list of run names
     runlist_wc - list of run names with wildcards
     varlist - list of variables to load
     
    examples:
    modellist= ['ACCESS-ESM1-5',  
                'CESM2',    
                'GFDL-ESM4',  
                'GISS_E2.1',  
                'NorESM2-LM',
                'MPI-ESM1-2-LR',
                'CNRM-ESM2-1',
                'HadCM3LC-Bris']
    runlist = ['flat10','flat10_zec','flat10_cdr']
    # use a wildcard to capture different ways the folders and runs are named across models
    runlist_wc = ['*lat10','*zec','*cdr']
    
    varlist_load=['cVeg','cSoil','cLitter','nbp','gpp','rh'] #, 'gpp','fgco2', 'ra', 'rh']#, 'npp'] # not working beyond nbp for norESM
    '''



    import numpy as np
    import numpy.matlib
    import numpy.ma as ma
    
    import xarray as xr
    #xr.set_options(enable_cftimeindex=True)
    #from xarray.coding.times import CFTimedeltaCoder
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True) #create time coder with cftime
    # CFTimedeltaCoder(decode_via_units=True)
    
    import time
    import cftime
    import netCDF4 as nc
    from datetime import timedelta
    
    import pandas as pd
    
    import glob
    
    
    ## notes on packages to add to this kernel
    import nc_time_axis





    # data location
    outputdir= '/glade/campaign/cgd/tss/people/aswann/flat10/'

    #----loop over models----#
    for m in range(len(modellist)):
    #for m in range(len(['GFDL-ESM4',  'GISS_E2.1',  'NorESM2-LM','MPI-ESM1-2-LR'])):
        model=modellist[m]
        print('loading model: ' +model)
        #----loop over experiments----# 
        for r in range(len(runlist)):
            run = runlist_wc[r]
            print('loading run: ' +run)
            #----loop over variables----#
            for v in range(len(varlist)):
                var=varlist[v]
                print('loading variable: ' +var)
                
                searchpath= outputdir +model +'/' +run +'/*' +var +'_*.nc'
                if ((model =='CESM2') or (model == 'UKESM1.2')):
                    # all models have a var_ filename except CESM
                    searchpath= outputdir +model +'/' +run +'/*' +var +'*.nc'
                
                filenamelist= np.sort(glob.glob(searchpath)) # sort in time order, xarray was having trouble arranging some of them in time dim
    
                #----loop over filenames----#
                # some variables are stored in multiple files
                # this should be possible with xr.open_mfdataset but it isn't loading all of time points
                for f in range(len(filenamelist)):
                    file = filenamelist[f]
                    if f==0:
                        dsmerge_f = xr.open_dataset(file,decode_times=time_coder,decode_timedelta=False)
                    else:
                        ds = xr.open_dataset(file,decode_times=time_coder,decode_timedelta=False)
                        dsmerge_f=xr.concat([dsmerge_f,ds],dim='time')
    
                #----- Dealing with GISS----#
                # GISS does not have a "time" index, and instead just a list of years
                # lets replace the "year" dimension (data is called "years")
                # with a cftime object called "time" so it matches the other models
                # some variables don't have the variable that defines years at all
                if ((model == 'GISS_E2.1') and ('time' not in dsmerge_f)):         
                    if 'year' in dsmerge_f: # if it has a variable called year, use that to make the time index
                        time_index = [cftime.DatetimeNoLeap(year, 1, 1) for year in dsmerge_f.year]
                    else: # if it does not have a variable for year, use the size of the year dimension to make the time index
                        startyear=[1850, 1950, 1950] # these are the start years for each experiment for GISS
                        years = np.arange(startyear[r], startyear[r]+len(dsmerge_f['year']))
                        time_index = [cftime.DatetimeNoLeap(year, 1, 1) for year in years]
                    
                    # Create a new DataArray with cftime objects
                    time_da = xr.DataArray(time_index, dims='year')
                    # Add time_da as a coordinate to the dataset
                    dsmerge_f.coords['time'] = time_da
                    # Now, swap dimensions from 'years' to 'time'
                    dsmerge_f = dsmerge_f.swap_dims({'year': 'time'})
                    # drop the year variable
                    #dsmerge_f = dsmerge_f.drop_vars('year')
                
                #----correct the name of the lat lon dimensions
                if (((model =='HadCM3LC-Bris') or (model == 'UKESM1.2')) and ('lat' not in dsmerge_f)):
                    #-- change latitude and longitude to lat and lon for HadCM3
                    dsmerge_f = dsmerge_f.rename({'longitude': 'lon','latitude': 'lat'})
                
                #----correct variable names----# 
                if 'nep' in dsmerge_f: # one model has nbp called nep instead -> add an nbp variable that is a copy of nep
                    dsmerge_f['nbp'] = dsmerge_f['nep']
                    #dsmerge_f = dsmerge_f.drop_vars('nep') # to remove it from the dataset
                
                if model =='HadCM3LC-Bris':
                    if 'GBMVegCarb_srf' in dsmerge_f: #HadCM3 
                        dsmerge_f['cVeg'] = dsmerge_f['GBMVegCarb_srf']
                    if 'soilCarbon_srf' in dsmerge_f: #HadCM3 
                        dsmerge_f['cSoil'] = dsmerge_f['soilCarbon_srf']
                    if 'NPP_mm_srf' in dsmerge_f: #HadCM3 
                        dsmerge_f['npp'] = dsmerge_f['NPP_mm_srf']
                    if 'unknown' in dsmerge_f: #HadCM3 
                        dsmerge_f['nbp'] = dsmerge_f['unknown']
                    if 'field1560_mm_srf' in dsmerge_f: #HadCM3 
                        dsmerge_f['fgco2'] = dsmerge_f['field1560_mm_srf']
                    if 'soilResp_mm_srf' in dsmerge_f: #HadCM3 cSoil
                        dsmerge_f['rh'] = dsmerge_f['soilResp_mm_srf']
                    if 'GPP_mm_srf gpp' in dsmerge_f: #HadCM3 cSoil
                        dsmerge_f['gpp'] = dsmerge_f['GPP_mm_srf gpp']
                    if 'temp_mm_1_5m' in dsmerge_f: #HadCM3 tas
                        dsmerge_f['tas']= dsmerge_f['temp_mm_1_5m']
                    if 'precip_mm_srf' in dsmerge_f: #HadCM3 pr
                        dsmerge_f['pr']= dsmerge_f['precip_mm_srf']
                     
                if model =='UKESM1.2':
                    missing_value = 1.0e36
                    for var_name, vari in dsmerge_f.data_vars.items(): #replace missing value with nan
                        # Apply only if variable is numeric and has at least one dimension
                        if np.issubdtype(vari.dtype, np.number):
                            try:
                                dsmerge_f[var_name] = vari.where(vari < missing_value * 0.1, np.nan)
                            except:
                                print('vari=' +str(vari) +' var_name=' +var_name)
                                raise
                    if 'vegetation_carbon_content' in dsmerge_f: #UKESM 
                        dsmerge_f['cVeg'] = dsmerge_f['vegetation_carbon_content']
                    if 'soil_carbon_content' in dsmerge_f: #UKESM
                        dsmerge_f['cSoil'] = dsmerge_f['soil_carbon_content']
                    if 'm01s19i102' in dsmerge_f: #UKESM 
                        dsmerge_f['npp'] = dsmerge_f['m01s19i102']
                    #if 'unknown' in dsmerge_f: #UKESM  ###I CANT FIND NBP, will need to be constructed from soil resp, plant resp, and gpp? should verify with UKESM group
                    #    dsmerge_f['nbp'] = dsmerge_f['unknown']
                    if 'm01s00i250' in dsmerge_f: #UKESM
                        dsmerge_f['fgco2'] = dsmerge_f['m01s00i250']
                    if 'm01s19i053' in dsmerge_f: #UKESM
                        dsmerge_f['rh'] = dsmerge_f['m01s19i053']
                    if 'm01s19i183' in dsmerge_f: #UKESM   
                        dsmerge_f['gpp'] = dsmerge_f['m01s19i183']
                    if 'air_temperature' in dsmerge_f: #UKESM
                        dsmerge_f['tas'] = dsmerge_f['air_temperature']
                    if 'precipitation_flux' in dsmerge_f: #UKESM
                        dsmerge_f['pr'] = dsmerge_f['precipitation_flux']

                
                if model == 'NorESM2-LM':
                    if 'PRECC' in dsmerge_f: #NorESM
                        dsmerge_f['pr']=dsmerge_f['PRECC']
                        if dsmerge_f['pr'].units == 'm/s':
                            dsmerge_f['pr']=dsmerge_f['pr']*(1e3)
                            dsmerge_f['pr'].attrs['units'] = 'kg m-2 s-1' #equivalent is mm/s
                    



                
                #----check units and convert if necessary----#
                if var in dsmerge_f: 
                    if model =='CESM2':
                        if dsmerge_f[var].units == 'gC/m^2/s':
                            dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
                            dsmerge_f[var].attrs['units'] = 'kg m-2 s-1'
                        # stock variables
                        elif dsmerge_f[var].units == 'gC/m^2':
                            dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
                            dsmerge_f[var].attrs['units'] = 'kg m-2'
    
                    # the units for cVeg in GISS look like they MUST be in gC rather than kgC 
                    # CHANGING THE UNIT - even though it is reported as kgC, assuming it is in gC
                    if ((var == 'cVeg') and (model == 'GISS_E2.1')):
                        dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
    
                    
                else: #var does not exist
                    ds=dsmerge_f
                    # add a blank variable so that loops work
                    if 'time' in ds:
                        nan_dataarray = xr.DataArray(np.full((len(ds['time']),len(ds['lat']), len(ds['lon'])), np.nan), 
                                                     coords={'lon': ds['lon'], 'lat': ds['lat'],'time': ds['time']}, dims=['time','lat', 'lon'])
                    #else: # this should now be obsolete
                    #    nan_dataarray = xr.DataArray(np.full((len(ds['year']),len(ds['lat']), len(ds['lon'])), np.nan), 
                    #             coords={'lon': ds['lon'], 'lat': ds['lat'],'year': ds['year']}, dims=['year','lat', 'lon'])
     
       
                    # Assign the new variable to the dataset
                    dsmerge_f[var] = nan_dataarray
                
                #----merge all variables into one dataset----#
                # if it's the first variable, then start a new datset, otherwise merge with existing
                if v ==0:
                    dsmerge_v = dsmerge_f.copy()
                else:
                    dsmerge_v=xr.merge([dsmerge_v, dsmerge_f],compat='override')
    
                # add a new variable that is the sum of all carbon pools
                if all(var_name in dsmerge_v for var_name in ['cVeg', 'cSoil', 'cLitter']):
                    if (dsmerge_v['cLitter'].notnull().all()): #litter is sometimes missing. Would be good to make this more general but dealing with this problem for now.
                        dsmerge_v['cTot'] = dsmerge_v['cVeg']+dsmerge_v['cSoil']+dsmerge_v['cLitter'] 
                    else: 
                        dsmerge_v['cTot'] = dsmerge_v['cVeg']+dsmerge_v['cSoil'] 
            
            #----save output to a dictionary----#
            print('adding ' +model +' ' +runlist[r] +' to dict')
            data_dict[model +'_' +runlist[r]] = dsmerge_v
    

    return data_dict


#==================
    
def load_one_model(model, run_wc, varlist):

    '''
     this function loads all variables in varlist for one experiment for one model
     into an xarray dataset
    
     model -  model names that match filenames
     run_wc - run name with wildcards
     varlist - list of variables to load
     
    examples:
    modellist= ['ACCESS-ESM1-5',  
                'CESM2',    
                'GFDL-ESM4',  
                'GISS_E2.1',  
                'NorESM2-LM',
                'MPI-ESM1-2-LR',
                'CNRM-ESM2-1',
                'HadCM3LC-Bris']
    runlist = ['flat10','flat10_zec','flat10_cdr']
    # use a wildcard to capture different ways the folders and runs are named across models
    runlist_wc = ['*lat10','*zec','*cdr']
    
    varlist_load=['cVeg','cSoil','cLitter','nbp','gpp','rh'] #, 'gpp','fgco2', 'ra', 'rh']#, 'npp'] # not working beyond nbp for norESM
    '''



    import numpy as np
    import numpy.matlib
    import numpy.ma as ma
    
    import xarray as xr
    #xr.set_options(enable_cftimeindex=True)
    #from xarray.coding.times import CFTimedeltaCoder
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True) #create time coder with cftime
    # CFTimedeltaCoder(decode_via_units=True)
    
    import time
    import cftime
    import netCDF4 as nc
    from datetime import timedelta
    
    import pandas as pd
    
    import glob
    
    
    ## notes on packages to add to this kernel
    import nc_time_axis





    # data location
    outputdir= '/glade/campaign/cgd/tss/people/aswann/flat10/'

        
    print('loading model: ' +model)

    run=run_wc
    print('loading run: ' +run)
    #----loop over variables----#
    for v in range(len(varlist)):
        var=varlist[v]
        print('loading variable: ' +var)
        
        searchpath= outputdir +model +'/' +run +'/*' +var +'_*.nc'
        if ((model =='CESM2') or (model == 'UKESM1.2')):
            # all models have a var_ filename except CESM
            searchpath= outputdir +model +'/' +run +'/*' +var +'*.nc'
        
        filenamelist= np.sort(glob.glob(searchpath)) # sort in time order, xarray was having trouble arranging some of them in time dim

        #----loop over filenames----#
        # some variables are stored in multiple files
        # this should be possible with xr.open_mfdataset but it isn't loading all of time points
        for f in range(len(filenamelist)):
            file = filenamelist[f]
            if f==0:
                dsmerge_f = xr.open_dataset(file,decode_times=time_coder,decode_timedelta=False)
            else:
                ds = xr.open_dataset(file,decode_times=time_coder,decode_timedelta=False)
                dsmerge_f=xr.concat([dsmerge_f,ds],dim='time')

        #----- Dealing with GISS----#
        # GISS does not have a "time" index, and instead just a list of years
        # lets replace the "year" dimension (data is called "years")
        # with a cftime object called "time" so it matches the other models
        # some variables don't have the variable that defines years at all
        if ((model == 'GISS_E2.1') and ('time' not in dsmerge_f)):         
            if 'year' in dsmerge_f: # if it has a variable called year, use that to make the time index
                time_index = [cftime.DatetimeNoLeap(year, 1, 1) for year in dsmerge_f.year]
            else: # if it does not have a variable for year, use the size of the year dimension to make the time index
                startyear=[1850, 1950, 1950] # these are the start years for each experiment for GISS
                years = np.arange(startyear[r], startyear[r]+len(dsmerge_f['year']))
                time_index = [cftime.DatetimeNoLeap(year, 1, 1) for year in years]
            
            # Create a new DataArray with cftime objects
            time_da = xr.DataArray(time_index, dims='year')
            # Add time_da as a coordinate to the dataset
            dsmerge_f.coords['time'] = time_da
            # Now, swap dimensions from 'years' to 'time'
            dsmerge_f = dsmerge_f.swap_dims({'year': 'time'})
            # drop the year variable
            #dsmerge_f = dsmerge_f.drop_vars('year')
        
        #----correct the name of the lat lon dimensions
        if (((model =='HadCM3LC-Bris') or (model == 'UKESM1.2')) and ('lat' not in dsmerge_f)):
            #-- change latitude and longitude to lat and lon for HadCM3
            dsmerge_f = dsmerge_f.rename({'longitude': 'lon','latitude': 'lat'})
        
        #----correct variable names----# 
        if 'nep' in dsmerge_f: # one model has nbp called nep instead -> add an nbp variable that is a copy of nep
            dsmerge_f['nbp'] = dsmerge_f['nep']
            #dsmerge_f = dsmerge_f.drop_vars('nep') # to remove it from the dataset
        
        if model =='HadCM3LC-Bris':
            if 'GBMVegCarb_srf' in dsmerge_f: #HadCM3 
                dsmerge_f['cVeg'] = dsmerge_f['GBMVegCarb_srf']
            if 'soilCarbon_srf' in dsmerge_f: #HadCM3 
                dsmerge_f['cSoil'] = dsmerge_f['soilCarbon_srf']
            if 'NPP_mm_srf' in dsmerge_f: #HadCM3 
                dsmerge_f['npp'] = dsmerge_f['NPP_mm_srf']
            if 'unknown' in dsmerge_f: #HadCM3 
                dsmerge_f['nbp'] = dsmerge_f['unknown']
            if 'field1560_mm_srf' in dsmerge_f: #HadCM3 
                dsmerge_f['fgco2'] = dsmerge_f['field1560_mm_srf']
            if 'soilResp_mm_srf' in dsmerge_f: #HadCM3 cSoil
                dsmerge_f['rh'] = dsmerge_f['soilResp_mm_srf']
            if 'GPP_mm_srf gpp' in dsmerge_f: #HadCM3 cSoil
                dsmerge_f['gpp'] = dsmerge_f['GPP_mm_srf gpp']
            if 'temp_mm_1_5m' in dsmerge_f: #HadCM3 tas
                dsmerge_f['tas']= dsmerge_f['temp_mm_1_5m']
            if 'precip_mm_srf' in dsmerge_f: #HadCM3 pr
                dsmerge_f['pr']= dsmerge_f['precip_mm_srf']
             
        if model =='UKESM1.2':
            missing_value = 1.0e36
            for var_name, vari in dsmerge_f.data_vars.items(): #replace missing value with nan
                # Apply only if variable is numeric and has at least one dimension
                if np.issubdtype(vari.dtype, np.number):
                    try:
                        dsmerge_f[var_name] = vari.where(vari < missing_value * 0.1, np.nan)
                    except:
                        print('vari=' +str(vari) +' var_name=' +var_name)
                        raise
            if 'vegetation_carbon_content' in dsmerge_f: #UKESM 
                dsmerge_f['cVeg'] = dsmerge_f['vegetation_carbon_content']
            if 'soil_carbon_content' in dsmerge_f: #UKESM
                dsmerge_f['cSoil'] = dsmerge_f['soil_carbon_content']
            if 'm01s19i102' in dsmerge_f: #UKESM 
                dsmerge_f['npp'] = dsmerge_f['m01s19i102']
            #if 'unknown' in dsmerge_f: #UKESM  ###I CANT FIND NBP, will need to be constructed from soil resp, plant resp, and gpp? should verify with UKESM group
            #    dsmerge_f['nbp'] = dsmerge_f['unknown']
            if 'm01s00i250' in dsmerge_f: #UKESM
                dsmerge_f['fgco2'] = dsmerge_f['m01s00i250']
            if 'm01s19i053' in dsmerge_f: #UKESM
                dsmerge_f['rh'] = dsmerge_f['m01s19i053']
            if 'm01s19i183' in dsmerge_f: #UKESM   
                dsmerge_f['gpp'] = dsmerge_f['m01s19i183']
            if 'air_temperature' in dsmerge_f: #UKESM
                dsmerge_f['tas'] = dsmerge_f['air_temperature']
            if 'precipitation_flux' in dsmerge_f: #UKESM
                dsmerge_f['pr'] = dsmerge_f['precipitation_flux']

        
        if model == 'NorESM2-LM':
            if 'PRECC' in dsmerge_f: #NorESM
                dsmerge_f['pr']=dsmerge_f['PRECC']
                if dsmerge_f['pr'].units == 'm/s':
                    dsmerge_f['pr']=dsmerge_f['pr']*(1e3)
                    dsmerge_f['pr'].attrs['units'] = 'kg m-2 s-1' #equivalent is mm/s
            



        
        #----check units and convert if necessary----#
        if var in dsmerge_f: 
            if model =='CESM2':
                if dsmerge_f[var].units == 'gC/m^2/s':
                    dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
                    dsmerge_f[var].attrs['units'] = 'kg m-2 s-1'
                # stock variables
                elif dsmerge_f[var].units == 'gC/m^2':
                    dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
                    dsmerge_f[var].attrs['units'] = 'kg m-2'

            # the units for cVeg in GISS look like they MUST be in gC rather than kgC 
            # CHANGING THE UNIT - even though it is reported as kgC, assuming it is in gC
            if ((var == 'cVeg') and (model == 'GISS_E2.1')):
                dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC

            
        else: #var does not exist
            ds=dsmerge_f
            # add a blank variable so that loops work
            if 'time' in ds:
                nan_dataarray = xr.DataArray(np.full((len(ds['time']),len(ds['lat']), len(ds['lon'])), np.nan), 
                                             coords={'lon': ds['lon'], 'lat': ds['lat'],'time': ds['time']}, dims=['time','lat', 'lon'])
            #else: # this should now be obsolete
            #    nan_dataarray = xr.DataArray(np.full((len(ds['year']),len(ds['lat']), len(ds['lon'])), np.nan), 
            #             coords={'lon': ds['lon'], 'lat': ds['lat'],'year': ds['year']}, dims=['year','lat', 'lon'])


            # Assign the new variable to the dataset
            dsmerge_f[var] = nan_dataarray
        
        #----merge all variables into one dataset----#
        # if it's the first variable, then start a new datset, otherwise merge with existing
        if v ==0:
            dsmerge_v = dsmerge_f.copy()
        else:
            dsmerge_v=xr.merge([dsmerge_v, dsmerge_f],compat='override')

        # add a new variable that is the sum of all carbon pools
        if all(var_name in dsmerge_v for var_name in ['cVeg', 'cSoil', 'cLitter']):
            if (dsmerge_v['cLitter'].notnull().all()): #litter is sometimes missing. Would be good to make this more general but dealing with this problem for now.
                dsmerge_v['cTot'] = dsmerge_v['cVeg']+dsmerge_v['cSoil']+dsmerge_v['cLitter'] 
            else: 
                dsmerge_v['cTot'] = dsmerge_v['cVeg']+dsmerge_v['cSoil'] 
    
    #----save output to a dictionary----#
    print('finished loading ' +model +' ' +run)
    #data_dict[model +'_' +runlist[r]] = dsmerge_v
    ## - save the matrix to a netcdf file
    ##dsmerge_v.to_netcdf(model +'_' +run +'.nc')

    return dsmerge_v

#==================
    
def load_one_model_onevar(model, run_wc, var):

    '''
     this function loads all variables in varlist for one experiment for one model
     into an xarray dataset
    
     model -  model names that match filenames
     run_wc - run name with wildcards
     varlist - list of variables to load
     
    examples:
    modellist= ['ACCESS-ESM1-5',  
                'CESM2',    
                'GFDL-ESM4',  
                'GISS_E2.1',  
                'NorESM2-LM',
                'MPI-ESM1-2-LR',
                'CNRM-ESM2-1',
                'HadCM3LC-Bris']
    runlist = ['flat10','flat10_zec','flat10_cdr']
    # use a wildcard to capture different ways the folders and runs are named across models
    runlist_wc = ['*lat10','*zec','*cdr']
    
    varlist_load=['cVeg','cSoil','cLitter','nbp','gpp','rh'] #, 'gpp','fgco2', 'ra', 'rh']#, 'npp'] # not working beyond nbp for norESM
    '''



    import numpy as np
    import numpy.matlib
    import numpy.ma as ma
    
    import xarray as xr
    #xr.set_options(enable_cftimeindex=True)
    #from xarray.coding.times import CFTimedeltaCoder
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True) #create time coder with cftime
    # CFTimedeltaCoder(decode_via_units=True)
    
    import time
    import cftime
    import netCDF4 as nc
    from datetime import timedelta
    
    import pandas as pd
    
    import glob
    
    
    ## notes on packages to add to this kernel
    import nc_time_axis


    speryr=60*60*24*365


    # data location
    outputdir= '/glade/campaign/cgd/tss/people/aswann/flat10/'

        
    #print('loading model: ' +model)

    run=run_wc
    #print('loading run: ' +run)

    print('loading variable: ' +var)
    
    searchpath= outputdir +model +'/' +run +'/*' +var +'_*.nc'
    if ((model =='CESM2') or (model == 'UKESM1.2')):
        # all models have a var_ filename except CESM
        searchpath= outputdir +model +'/' +run +'/*' +var +'*.nc'
    if (model =='NorESM2-LM'):
        # all models have a var_ filename except CESM
        searchpath= outputdir +model +'/' +run +'/' +var +'_*.nc'
    
    filenamelist= np.sort(glob.glob(searchpath)) # sort in time order, xarray was having trouble arranging some of them in time dim

    # initialize
    dsmerge_f=None
    
    #----loop over filenames----#
    # some variables are stored in multiple files
    # this should be possible with xr.open_mfdataset but it isn't loading all of time points
    for f in range(len(filenamelist)):
        file = filenamelist[f]
        if f==0:
            dsmerge_f = xr.open_dataset(file,decode_times=time_coder,decode_timedelta=False, chunks={'time': 10})
        else:
            ds = xr.open_dataset(file,decode_times=time_coder,decode_timedelta=False, chunks={'time': 10})
            dsmerge_f=xr.concat([dsmerge_f,ds],dim='time')

    if dsmerge_f is None: 
        print('no data for ' +var)
        return # exit this function
        
    #----- Dealing with GISS----#
    # GISS does not have a "time" index, and instead just a list of years
    # lets replace the "year" dimension (data is called "years")
    # with a cftime object called "time" so it matches the other models
    # some variables don't have the variable that defines years at all
    if ((model == 'GISS_E2.1') and ('time' not in dsmerge_f)):         
        if 'year' in dsmerge_f: # if it has a variable called year, use that to make the time index
            time_index = [cftime.DatetimeNoLeap(year, 1, 1) for year in dsmerge_f.year]
        else: # if it does not have a variable for year, use the size of the year dimension to make the time index
            #runlist_wc = ['*lat10','*zec','*cdr']
            # startyear=[1850, 1950, 1950] # these are the start years for each experiment for GISS
            if run=='*lat10':
                startyear=1850
            elif run=='*zec':
                startyear=1950
            elif run=='*cdr':
                startyear=1950
            years = np.arange(startyear, startyear+len(dsmerge_f['year']))
            time_index = [cftime.DatetimeNoLeap(year, 1, 1) for year in years]
        
        # Create a new DataArray with cftime objects
        time_da = xr.DataArray(time_index, dims='year')
        # Add time_da as a coordinate to the dataset
        dsmerge_f.coords['time'] = time_da
        # Now, swap dimensions from 'years' to 'time'
        dsmerge_f = dsmerge_f.swap_dims({'year': 'time'})
        # drop the year variable
        #dsmerge_f = dsmerge_f.drop_vars('year')
    
    #----correct the name of the lat lon dimensions
    if (((model =='HadCM3LC-Bris') or (model == 'UKESM1.2')) and ('lat' not in dsmerge_f)):
        #-- change latitude and longitude to lat and lon for HadCM3
        dsmerge_f = dsmerge_f.rename({'longitude': 'lon','latitude': 'lat'})
    
    #----correct variable names----# 
    if 'nep' in dsmerge_f: # one model has nbp called nep instead -> add an nbp variable that is a copy of nep
        dsmerge_f['nbp'] = dsmerge_f['nep']
        #dsmerge_f = dsmerge_f.drop_vars('nep') # to remove it from the dataset
    
    if model =='HadCM3LC-Bris':
        if 'GBMVegCarb_srf' in dsmerge_f: #HadCM3 
            dsmerge_f['cVeg'] = dsmerge_f['GBMVegCarb_srf']
        if 'soilCarbon_srf' in dsmerge_f: #HadCM3 
            dsmerge_f['cSoil'] = dsmerge_f['soilCarbon_srf']
        if 'NPP_mm_srf' in dsmerge_f: #HadCM3 
            dsmerge_f['npp'] = dsmerge_f['NPP_mm_srf']
        if 'unknown' in dsmerge_f: #HadCM3 
            dsmerge_f['nbp'] = dsmerge_f['unknown']
        if 'field1560_mm_srf' in dsmerge_f: #HadCM3 
            dsmerge_f['fgco2'] = dsmerge_f['field1560_mm_srf']
        if 'soilResp_mm_srf' in dsmerge_f: #HadCM3 cSoil
            dsmerge_f['rh'] = dsmerge_f['soilResp_mm_srf']
        if 'GPP_mm_srf gpp' in dsmerge_f: #HadCM3 cSoil
            dsmerge_f['gpp'] = dsmerge_f['GPP_mm_srf gpp']
        if 'temp_mm_1_5m' in dsmerge_f: #HadCM3 tas
            dsmerge_f['tas']= dsmerge_f['temp_mm_1_5m']
        if 'precip_mm_srf' in dsmerge_f: #HadCM3 pr
            dsmerge_f['pr']= dsmerge_f['precip_mm_srf']
         
    if model =='UKESM1.2':
        missing_value = 1.0e36
        for var_name, vari in dsmerge_f.data_vars.items(): #replace missing value with nan
            # Apply only if variable is numeric and has at least one dimension
            if np.issubdtype(vari.dtype, np.number):
                try:
                    dsmerge_f[var_name] = vari.where(vari < missing_value * 0.1, np.nan)
                except:
                    print('vari=' +str(vari) +' var_name=' +var_name)
                    raise
        if 'vegetation_carbon_content' in dsmerge_f: #UKESM 
            dsmerge_f['cVeg'] = dsmerge_f['vegetation_carbon_content']
        if 'soil_carbon_content' in dsmerge_f: #UKESM
            dsmerge_f['cSoil'] = dsmerge_f['soil_carbon_content']
        if 'm01s19i102' in dsmerge_f: #UKESM 
            dsmerge_f['npp'] = dsmerge_f['m01s19i102']*(1/speryr)
        #if 'unknown' in dsmerge_f: #UKESM  ###I CANT FIND NBP, will need to be constructed from soil resp, plant resp, and gpp? should verify with UKESM group
        #    dsmerge_f['nbp'] = dsmerge_f['unknown']
        if 'm01s00i250' in dsmerge_f: #UKESM
            dsmerge_f['fgco2'] = dsmerge_f['m01s00i250']
        if 'm01s19i053' in dsmerge_f: #UKESM
            dsmerge_f['rh'] = dsmerge_f['m01s19i053']*(1/speryr)
        if 'm01s19i183' in dsmerge_f: #UKESM   
            dsmerge_f['gpp'] = dsmerge_f['m01s19i183']*(1/speryr)
        if 'air_temperature' in dsmerge_f: #UKESM
            dsmerge_f['tas'] = dsmerge_f['air_temperature']
        if 'precipitation_flux' in dsmerge_f: #UKESM
            dsmerge_f['pr'] = dsmerge_f['precipitation_flux']
        if 'nbp' in dsmerge_f: #UKESM
            dsmerge_f['nbp'] = dsmerge_f['nbp']*(1/speryr)
        # if we were loading all variables at once this would work, but one at a time it does not. Instead I made new nc files that did this calculation already
        # if model =='UKESM1.2': #UKESM is missing nbp, but it can be calculated from npp and rh
        #NBP = NPP-RH
        # if ('npp' in dsmerge_f) and ('rh' in dsmerge_f):
        #     dsmerge_f['nbp'] = dsmerge_f['npp'] - dsmerge_f['rh']

    
    if model == 'NorESM2-LM':
        if 'PRECC' in dsmerge_f: #NorESM
            dsmerge_f['pr']=dsmerge_f['PRECC']
            if dsmerge_f['pr'].units == 'm/s':
                dsmerge_f['pr']=dsmerge_f['pr']*(1e3)
                dsmerge_f['pr'].attrs['units'] = 'kg m-2 s-1' #equivalent is mm/s
        



    
    #----check units and convert if necessary----#
    if var in dsmerge_f: 
        if model =='CESM2':
            if dsmerge_f[var].units == 'gC/m^2/s':
                dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
                dsmerge_f[var].attrs['units'] = 'kg m-2 s-1'
            # stock variables
            elif dsmerge_f[var].units == 'gC/m^2':
                dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC
                dsmerge_f[var].attrs['units'] = 'kg m-2'

        # the units for cVeg in GISS look like they MUST be in gC rather than kgC 
        # CHANGING THE UNIT - even though it is reported as kgC, assuming it is in gC
        if ((var == 'cVeg') and (model == 'GISS_E2.1')):
            dsmerge_f[var]=dsmerge_f[var]*(1/1000) # convert from gC to kgC

        
    else: #var does not exist
        ds=dsmerge_f
        # add a blank variable so that loops work
        if 'time' in ds:
            nan_dataarray = xr.DataArray(np.full((len(ds['time']),len(ds['lat']), len(ds['lon'])), np.nan), 
                                         coords={'lon': ds['lon'], 'lat': ds['lat'],'time': ds['time']}, dims=['time','lat', 'lon'])
        #else: # this should now be obsolete
        #    nan_dataarray = xr.DataArray(np.full((len(ds['year']),len(ds['lat']), len(ds['lon'])), np.nan), 
        #             coords={'lon': ds['lon'], 'lat': ds['lat'],'year': ds['year']}, dims=['year','lat', 'lon'])


        # Assign the new variable to the dataset
        dsmerge_f[var] = nan_dataarray
    


    
    #----save output to a dictionary----#
    print('finished loading ' +model +' ' +run +' ' +var)
    #data_dict[model +'_' +runlist[r]] = dsmerge_v
    ## - save the matrix to a netcdf file
    ##dsmerge_v.to_netcdf(model +'_' +run +'.nc')

    return dsmerge_f



#==================
def load_grid(data_dict,modellist):
    '''
    this function loads the grid information including area of cells, ocean area, and land fraction
    
    data_dict - dictionary to hold xarrays
    modellist - list array of model names that match filenames
     
    examples:
    modellist= ['ACCESS-ESM1-5',  
                'CESM2',    
                'GFDL-ESM4',  
                'GISS_E2.1',  
                'NorESM2-LM',
                'MPI-ESM1-2-LR',
                'CNRM-ESM2-1',
                'HadCM3LC-Bris']

    '''

    import numpy as np
    import numpy.matlib
    import numpy.ma as ma
    
    import xarray as xr
    #xr.set_options(enable_cftimeindex=True)
    #from xarray.coding.times import CFTimedeltaCoder
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True) #create time coder with cftime
    
    import time
    import cftime
    import netCDF4 as nc
    from datetime import timedelta
    
    import pandas as pd
    
    import glob
    
    
    ## notes on packages to add to this kernel
    import nc_time_axis

    # data location
    outputdir= '/glade/campaign/cgd/tss/people/aswann/flat10/'
    
    # loop over models
    for m in range(len(modellist)):
        model=modellist[m]
    
        print(model +' getting grid info')
        # get land fraction
        filenamelist= glob.glob(outputdir +model +'/*/*sftlf*.nc')
        landfrac = xr.open_dataset(filenamelist[0],decode_times=time_coder)
    
        # get area of gridcells
        filenamelist= glob.glob(outputdir +model +'/*/*areacella*.nc')
        areacella = xr.open_dataset(filenamelist[0],decode_times=time_coder)
       
        #----correct the name of the lat lon dimensions for landfrac and areacella
        if (((model =='HadCM3LC-Bris') or (model == 'UKESM1.2')) and ('lat' not in landfrac)):
            #-- change latitude and longitude to lat and lon for HadCM3
            landfrac = landfrac.rename({'longitude': 'lon','latitude': 'lat'})
            areacella = areacella.rename({'longitude': 'lon','latitude': 'lat'})
            #-- change name of area fields to match other models
            areacella = areacella.rename({'cell_area': 'areacella'})
            landfrac = landfrac.rename({'land_area_fraction': 'sftlf'})
    
        if (model =='GISS_E2.1'):
            # lon is -180 to 180 in data but 0 to 360 in grid files =>convert
            areacella['lon']=areacella['lon']-180
            landfrac['lon']=landfrac['lon']-180
            #landfrac['lon']=landfrac['lon']-180
            #landfrac.reindex_like(areacella, method='nearest',tolerance=0.05)
            
        # add to the dictionary
        data_dict[model +'_areacella'] = areacella
        data_dict[model +'_landfrac'] = landfrac

       ## get area of ocean gridcells
       # filenamelist= glob.glob(outputdir +model +'/*/*areacello*.nc')
       # areacello = xr.open_dataset(filenamelist[0],decode_times=time_coder)
        
       # if model =='CESM2':
       #     areacello=areacello*1e-4 # CESM2 has area units of cm2 for ocean
    
       # data_dict[model +'_areacello'] = areacello
    
    return data_dict




#==================
def weighted_temporal_mean(ds, var):
    """
    takes an annual average weighted by days in each month

    Args:
    - dataset: xarray dataset with monthly resolution data
    - var: variable name to be averaged

    Returns:
    - the weighted average

    Example Usage: 
    
    """

    import numpy as np
    import numpy.matlib
    import numpy.ma as ma
    
    import xarray as xr
    #xr.set_options(enable_cftimeindex=True)
    #from xarray.coding.times import CFTimedeltaCoder

    import time
    import cftime
    
    ## notes on packages to add to this kernel
    import nc_time_axis
    # Determine the month length
    month_length = ds.time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("time.year") / month_length.groupby("time.year").sum()

    # Make sure the weights in each year add up to 1
    np.testing.assert_allclose(wgts.groupby("time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(time="YS").sum(dim="time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(time="YS").sum(dim="time")

    # Return the weighted average
    return obs_sum / ones_out


#=================
def select_time_slice(dataset, startyear, endyear):
    """
    Selects the years between start year and end year
    of data from an xarray dataset.
    start year references 0 as the first year
    
    Args:
    - dataset: xarray dataset containing the time dimension
    
    Returns:
    - time_slice: xarray dataset containing the years of data between startyear and endyear

    Example usage:
    Assuming you have a list of xarray datasets called 'datasets'
    first_10_years_datasets = [select_first_10_years(dataset,startyear,endyear) for dataset in datasets]
    alternatively just retrieve a time slice for one dataset at a time
    """

    import numpy as np
    import numpy.matlib
    import numpy.ma as ma
    
    import xarray as xr
    #xr.set_options(enable_cftimeindex=True)
    #from xarray.coding.times import CFTimedeltaCoder
    
    import time
    import cftime

    # GISS originally did not have a cftime index. I now add one when data is loaded for GISS
    # This part of the loop should now be obselete, suggest deleting later.
    # Check if the dataset has a dimension labeled "time"
    if 'time' not in dataset.dims:
        print('time is not defined as cftime in this dataset')
        # if the model is GISS, then it does not have a cftime time index 
        # and instead just an index of years
        # also the first many enties of time are empty, so force the start date of 1850
        time_coord = dataset.year.values
        time_slice = dataset.sel(year=slice(1850 + startyear, 1850 + endyear))

    else:
        # model uses cftime
        # Extract the time coordinate from the dataset
        time_coord = dataset.time.values
        
        # Determine the calendar type of the dataset
        calendar_type = time_coord[0].calendar
        
        # Calculate the end date for the first 10 years based on the calendar type
        if calendar_type in ['standard', 'gregorian', 'proleptic_gregorian']:
            start_date = cftime.DatetimeGregorian(time_coord[0].year + startyear, time_coord[0].month, time_coord[0].day)
            end_date = cftime.DatetimeGregorian(time_coord[0].year + endyear, time_coord[0].month, time_coord[0].day)
        elif calendar_type == 'noleap':
            start_date = cftime.DatetimeNoLeap(time_coord[0].year + startyear, time_coord[0].month, time_coord[0].day)
            end_date = cftime.DatetimeNoLeap(time_coord[0].year + endyear, time_coord[0].month, time_coord[0].day)
        elif calendar_type in ['365_day','360_day']:
            start_date = cftime.Datetime360Day(time_coord[0].year + startyear, time_coord[0].month, time_coord[0].day)
            end_date = cftime.Datetime360Day(time_coord[0].year + endyear, time_coord[0].month, time_coord[0].day)
        # Add more conditions for other calendar types if needed
        
        # Select the time slice
        time_slice = dataset.sel(time=slice(start_date, end_date))
    
    return time_slice

#==================
    
def load_observations(zonal):

    '''
     this function loads observational datasets

     inputs: flag that indicates output for zonal mean
     zonal = 0 full grid output only
     zonal = 1 also output zonal mean
     
     outputs:
     cSoil, cVeg, cSoil_zonal, cVeg_zonal
     
    '''

    import numpy as np
    # import numpy.matlib
    # import numpy.ma as ma
    
    import xarray as xr
    # time_coder = xr.coders.CFDatetimeCoder(use_cftime=True) #create time coder with cftime
    
    # import time
    # import cftime
    # import netCDF4 as nc
    # from datetime import timedelta

    # cVeg
    # XuSaatchi.nc

    #-------- load data
    outputdir= '/glade/work/aswann/datasets/'

    cSoil=xr.open_dataset(outputdir +'cSoil_fx_HWSD2_19600101-20220101.nc') 
    cVeg=xr.open_dataset(outputdir +'XuSaatchi.nc')

    # ------- convert the longitude grid to 0 to 360
    # soil
    unitconvert_soil= 1e-12 #convert from kgC to PgC
    soilCdb=cSoil.copy(deep=True)
    # convert longitude 
    lon=soilCdb['lon'].values
    lon360=np.where(lon<0,lon +360,lon)
    soilCdb['lon']=lon360 # convert lon to 0-360
    soilCdb = soilCdb.sortby(soilCdb.lon)
    cSoil=soilCdb['cSoil']*unitconvert_soil # extract the variable and convert to PgC/m2
    
    # veg
    unitconvert_veg= 1e-13#Mg/ha to Pg/m2, 1ha/m2=1e-4, Mg/Pg = 1e6/1e15 =1e-9 => 1e-13
    vegCdb=cVeg.copy(deep=True)
    # convert longitude 
    lon=vegCdb['lon'].values
    lon360=np.where(lon<0,lon +360,lon)
    vegCdb['lon']=lon360 # convert lon to 0-360 
    vegCdb = vegCdb.sortby(vegCdb.lon)
    cVeg=vegCdb['biomass']*unitconvert_veg # extract the variable and convert to PgC/m2

    if zonal==0:
        print('full grid output only')
        return cSoil, cVeg
    else:
        print('creating zonal average output')
        #------- load area and land fraction from a model so we can area weight
        from loading_function_flat10 import  load_grid
    
        # load grid info for the highest resolution grid which we will interpolate
        outputdir= '/glade/campaign/cgd/tss/people/aswann/flat10/'
        modellist=['CESM2']
        # initialize a dictionary to hold grid data
        data_dict={}
        data_dict = load_grid(data_dict,modellist)
    
        model='CESM2' # this is the highest resolution model
    
        #--- get area and land fraction
        ds_area = data_dict[model +'_' +'areacella']
        ds_landfrac = data_dict[model +'_' +'landfrac']
    
        #--------- calculate zonal profiles
        # interpolate to the cSoil grid
        area = ds_area['areacella'].squeeze().interp_like(soilCdb, method='nearest')
        landfrac=ds_landfrac['sftlf'].interp_like(soilCdb, method='nearest')
        cSoil_zonal=(cSoil*landfrac*area).sum(dim='lon').squeeze() 

        # interpolate to the cVeg grid
        area = ds_area['areacella'].squeeze().interp_like(vegCdb, method='nearest')
        landfrac=ds_landfrac['sftlf'].interp_like(vegCdb, method='nearest')
        cVeg_zonal=(cVeg*landfrac*area).sum(dim='lon').mean(dim='time').squeeze()
    
        return cSoil, cVeg, cSoil_zonal, cVeg_zonal