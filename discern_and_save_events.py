#!/usr/bin/env python
# coding: utf-8

# In[1]:


# this script takes the time list from discover_events 
# and applies that list to filter the full ERA5 record down to only enhanced IVT events.


# In[1]:


import math
import xarray as xr
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta


# In[2]:


#parent_dir = "/nfs/turbo/clasp-aepayne/ERA5/rendfrey/southern_hemisphere/"

parent_dir = "/raid01/rendfrey/IVT/"
ivt_east_dir = "/raid01/rendfrey/IVT_and_MSLP/vertical_integral_of_eastward_water_vapour_flux/"
ivt_north_dir = "/raid01/rendfrey/IVT_and_MSLP/vertical_integral_of_northward_water_vapour_flux/"
mslp_dir = "/raid01/rendfrey/IVT_and_MSLP/mslp/"

monthly_mean_dir = "/raid01/rendfrey/transfers/monthly_means/"

# In[3]:


basin_set = [21, 22]
#basin_set = [9, 11]
#basin_set = [10]

seasons = ["DJF", "MAM", "JJA", "SON"]

percentile_cutoff = 90
threshold = percentile_cutoff / 100.


# In[4]:


#ds_ars = xr.open_mfdataset("/nfs/turbo/clasp-aepayne/ERA5/rendfrey/southern_hemisphere/AR_tags/cs_era/*")


# In[5]:


# open IVT, MSLP files

# open IVT files
ivt_ds_DJF = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[0]+".nc") # total IVT calculated by a previous script
ivt_ds_MAM = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[1]+".nc") # total IVT calculated by a previous script
ivt_ds_JJA = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[2]+".nc") # total IVT calculated by a previous script
ivt_ds_SON = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[3]+".nc") # total IVT calculated by a previous script

ivt_east_ds = xr.open_mfdataset(ivt_east_dir+"*.nc") # IVT east component from ERA5
ivt_north_ds = xr.open_mfdataset(ivt_north_dir+"*.nc") # IVT north component from ERA5

# open monthly IVT means
ivt_monthly_means = xr.open_dataset(monthly_mean_dir+"1979-2021_monthly_mean_moisture_flux.nc") # from ERA5

# convert 0, 360 longitude to -180, 180 in the monthly means
ivt_monthly_means.coords['longitude'] = (ivt_monthly_means.coords['longitude'] + 180) % 360 - 180
ivt_monthly_means = ivt_monthly_means.sortby(ivt_monthly_means.longitude)

# open MSLP files
mslp_ds = xr.open_mfdataset(mslp_dir+"*.nc") # mean sea level pressure

# open monthly MSLP mean
mslp_monthly_means = xr.open_dataset(monthly_mean_dir+"1979-2021_monthly_average_mean_sea_level_pressure.nc") # from ERA5

# convert 0, 360 longitude to -180, 180 in the monthly means
mslp_monthly_means.coords['longitude'] = (mslp_monthly_means.coords['longitude'] + 180) % 360 - 180
mslp_monthly_means = mslp_monthly_means.sortby(mslp_monthly_means.longitude)


# In[6]:


def get_season(ds, season):
    ds = ds.sel(time=(ds['time.season']==season))
    return ds


# In[7]:


# total IVT and its percentiles are already calculated by season, but the rest of the data files are not
# therefore, cut the relevant season out from the files

ivt_east_ds_DJF = get_season(ivt_east_ds, seasons[0])
ivt_north_ds_DJF = get_season(ivt_north_ds, seasons[0])
ivt_monthly_means_DJF = get_season(ivt_monthly_means, seasons[0])

mslp_ds_DJF = get_season(mslp_ds, seasons[0])
mslp_monthly_means_DJF = get_season(mslp_monthly_means, seasons[0])


# In[8]:


# total IVT and its percentiles are already calculated by season, but the rest of the data files are not
# therefore, cut the relevant season out from the files

ivt_east_ds_MAM = get_season(ivt_east_ds, seasons[1])
ivt_north_ds_MAM = get_season(ivt_north_ds, seasons[1])
ivt_monthly_means_MAM = get_season(ivt_monthly_means, seasons[1])

mslp_ds_MAM = get_season(mslp_ds, seasons[1])
mslp_monthly_means_MAM = get_season(mslp_monthly_means, seasons[1])


# In[9]:


ivt_east_ds_JJA = get_season(ivt_east_ds, seasons[2])
ivt_north_ds_JJA = get_season(ivt_north_ds, seasons[2])
ivt_monthly_means_JJA = get_season(ivt_monthly_means, seasons[2])

mslp_ds_JJA = get_season(mslp_ds, seasons[2])
mslp_monthly_means_JJA = get_season(mslp_monthly_means, seasons[2])


# In[10]:


ivt_east_ds_SON = get_season(ivt_east_ds, seasons[3])
ivt_north_ds_SON = get_season(ivt_north_ds, seasons[3])
ivt_monthly_means_SON = get_season(ivt_monthly_means, seasons[3])

mslp_ds_SON = get_season(mslp_ds, seasons[3])
mslp_monthly_means_SON = get_season(mslp_monthly_means, seasons[3])


# In[11]:


# take the time average of the monthly means

ivt_monthly_means_DJF = ivt_monthly_means_DJF.mean(dim="time")

mslp_monthly_means_DJF = mslp_monthly_means_DJF.mean(dim="time")


# In[12]:


# take the time average of the monthly means

ivt_monthly_means_MAM = ivt_monthly_means_MAM.mean(dim="time")

mslp_monthly_means_MAM = mslp_monthly_means_MAM.mean(dim="time")


# In[13]:


# take the time average of the monthly means

ivt_monthly_means_JJA = ivt_monthly_means_JJA.mean(dim="time")

mslp_monthly_means_JJA = mslp_monthly_means_JJA.mean(dim="time")


# In[14]:


# take the time average of the monthly means

ivt_monthly_means_SON = ivt_monthly_means_SON.mean(dim="time")

mslp_monthly_means_SON = mslp_monthly_means_SON.mean(dim="time")


# In[15]:


# combine datasets of the eastward and northward WV transport components
ivt_combined_ds_DJF = xr.merge([ivt_east_ds_DJF, ivt_north_ds_DJF])


# In[16]:


# combine datasets of the eastward and northward WV transport components
ivt_combined_ds_MAM = xr.merge([ivt_east_ds_MAM, ivt_north_ds_MAM])


# In[17]:


# combine datasets of the eastward and northward WV transport components
ivt_combined_ds_JJA = xr.merge([ivt_east_ds_JJA, ivt_north_ds_JJA])


# In[18]:


# combine datasets of the eastward and northward WV transport components
ivt_combined_ds_SON = xr.merge([ivt_east_ds_SON, ivt_north_ds_SON])


# In[19]:


# compute data arrays of IVT anomaly components
da_ivt_eastward_anomaly_DJF = ivt_combined_ds_DJF["p71.162"] - ivt_monthly_means_DJF["p71.162"]
da_ivt_northward_anomaly_DJF = ivt_combined_ds_DJF["p72.162"] - ivt_monthly_means_DJF["p72.162"]

# merge ivt anomaly components into one dataset 
ivt_combined_anomaly_ds_DJF = xr.merge([da_ivt_eastward_anomaly_DJF, da_ivt_northward_anomaly_DJF])

# calculate ivt anomaly magnitude
da_ivt_anomaly_DJF = np.sqrt(da_ivt_northward_anomaly_DJF**2 + da_ivt_eastward_anomaly_DJF**2)
da_ivt_anomaly_DJF.attrs["units"] = "kg m^-1 s^-1"

# compute data array of the mslp anomaly
da_mslp_anomaly_DJF = mslp_ds_DJF["msl"] - mslp_monthly_means_DJF["msl"]

# set mslp units to hPa
da_mslp_DJF = mslp_ds_DJF["msl"] / 100.
da_mslp_anomaly_DJF = da_mslp_anomaly_DJF / 100.


# In[20]:


# compute data arrays of IVT anomaly components
da_ivt_eastward_anomaly_MAM = ivt_combined_ds_MAM["p71.162"] - ivt_monthly_means_MAM["p71.162"]
da_ivt_northward_anomaly_MAM = ivt_combined_ds_MAM["p72.162"] - ivt_monthly_means_MAM["p72.162"]

# merge ivt anomaly components into one dataset 
ivt_combined_anomaly_ds_MAM = xr.merge([da_ivt_eastward_anomaly_MAM, da_ivt_northward_anomaly_MAM])

# calculate ivt anomaly magnitude
da_ivt_anomaly_MAM = np.sqrt(da_ivt_northward_anomaly_MAM**2 + da_ivt_eastward_anomaly_MAM**2)
da_ivt_anomaly_MAM.attrs["units"] = "kg m^-1 s^-1"

# compute data array of the mslp anomaly
da_mslp_anomaly_MAM = mslp_ds_MAM["msl"] - mslp_monthly_means_MAM["msl"]

# set mslp units to hPa
da_mslp_MAM = mslp_ds_MAM["msl"] / 100.
da_mslp_anomaly_MAM = da_mslp_anomaly_MAM / 100.


# In[21]:


# compute data arrays of IVT anomaly components
da_ivt_eastward_anomaly_JJA = ivt_combined_ds_JJA["p71.162"] - ivt_monthly_means_JJA["p71.162"]
da_ivt_northward_anomaly_JJA = ivt_combined_ds_JJA["p72.162"] - ivt_monthly_means_JJA["p72.162"]

# merge ivt anomaly components into one dataset 
ivt_combined_anomaly_ds_JJA = xr.merge([da_ivt_eastward_anomaly_JJA, da_ivt_northward_anomaly_JJA])

# calculate ivt anomaly magnitude
da_ivt_anomaly_JJA = np.sqrt(da_ivt_northward_anomaly_JJA**2 + da_ivt_eastward_anomaly_JJA**2)
da_ivt_anomaly_JJA.attrs["units"] = "kg m^-1 s^-1"

# compute data array of the mslp anomaly
da_mslp_anomaly_JJA = mslp_ds_JJA["msl"] - mslp_monthly_means_JJA["msl"]

# set mslp units to hPa
da_mslp_JJA = mslp_ds_JJA["msl"] / 100.
da_mslp_anomaly_JJA = da_mslp_anomaly_JJA / 100.


# In[22]:


# compute data arrays of IVT anomaly components
da_ivt_eastward_anomaly_SON = ivt_combined_ds_SON["p71.162"] - ivt_monthly_means_SON["p71.162"]
da_ivt_northward_anomaly_SON = ivt_combined_ds_SON["p72.162"] - ivt_monthly_means_SON["p72.162"]

# merge ivt anomaly components into one dataset 
ivt_combined_anomaly_ds_SON = xr.merge([da_ivt_eastward_anomaly_SON, da_ivt_northward_anomaly_SON])

# calculate ivt anomaly magnitude
da_ivt_anomaly_SON = np.sqrt(da_ivt_northward_anomaly_SON**2 + da_ivt_eastward_anomaly_SON**2)
da_ivt_anomaly_SON.attrs["units"] = "kg m^-1 s^-1"

# compute data array of the mslp anomaly
da_mslp_anomaly_SON = mslp_ds_SON["msl"] - mslp_monthly_means_SON["msl"]

# set mslp units to hPa
da_mslp_SON = mslp_ds_SON["msl"] / 100.
da_mslp_anomaly_SON = da_mslp_anomaly_SON / 100.


# In[23]:


# get data set filtered down to only the time steps with interesting IVT over the chosen glacier basin
# the following combines two event lists
if len(basin_set) == 2:

    # get all "A" basin times
    filtered_ds_A_DJF = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[0]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_A_DJF = filtered_ds_A_DJF.time.values

    # get all "B" basin times
    filtered_ds_B_DJF = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[0]+"_on_basin_"+str(basin_set[1])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_B_DJF = filtered_ds_B_DJF.time.values

    # following line says "what does the left set have that the right does not?"
    times_B_excl_A_DJF = set(timelist_B_DJF) - set(timelist_A_DJF) # times in B that are not in A

    times_B_excl_A_array_DJF = np.array(list(times_B_excl_A_DJF)) # change set into array
    times_A_and_B_DJF = np.concatenate((times_B_excl_A_array_DJF, timelist_A_DJF), axis=0) # combine B and A time coordinates

    big_list_DJF = times_A_and_B_DJF[:(len(times_B_excl_A_array_DJF)+len(timelist_A_DJF)-1)]
    all_times_DJF = np.sort(big_list_DJF)
    event_timelist_DJF = all_times_DJF

else:
    filtered_ds_DJF = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[0]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    event_timelist_DJF = filtered_ds_DJF.time.values 


# In[24]:


# get data set filtered down to only the time steps with interesting IVT over the chosen glacier basin
# the following combines two event lists
if len(basin_set) == 2:

    # get all "A" basin times
    filtered_ds_A_MAM = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[1]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_A_MAM = filtered_ds_A_MAM.time.values

    # get all "B" basin times
    filtered_ds_B_MAM = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[1]+"_on_basin_"+str(basin_set[1])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_B_MAM = filtered_ds_B_MAM.time.values

    # following line says "what does the left set have that the right does not?"
    times_B_excl_A_MAM = set(timelist_B_MAM) - set(timelist_A_MAM) # times in B that are not in A

    times_B_excl_A_array_MAM = np.array(list(times_B_excl_A_MAM)) # change set into array
    times_A_and_B_MAM = np.concatenate((times_B_excl_A_array_MAM, timelist_A_MAM), axis=0) # combine B and A time coordinates

    big_list_MAM = times_A_and_B_MAM[:(len(times_B_excl_A_array_MAM)+len(timelist_A_MAM)-1)]
    all_times_MAM = np.sort(big_list_MAM)
    event_timelist_MAM = all_times_MAM
    
else:
    filtered_ds_MAM = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[1]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    event_timelist_MAM = filtered_ds_MAM.time.values 


# In[25]:


# get data set filtered down to only the time steps with interesting IVT over the chosen glacier basin
# the following combines two event lists
if len(basin_set) == 2:

    # get all "A" basin times
    filtered_ds_A_JJA = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[2]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_A_JJA = filtered_ds_A_JJA.time.values

    # get all "B" basin times
    filtered_ds_B_JJA = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[2]+"_on_basin_"+str(basin_set[1])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_B_JJA = filtered_ds_B_JJA.time.values

    # following line says "what does the left set have that the right does not?"
    times_B_excl_A_JJA = set(timelist_B_JJA) - set(timelist_A_JJA) # times in B that are not in A

    times_B_excl_A_array_JJA = np.array(list(times_B_excl_A_JJA)) # change set into array
    times_A_and_B_JJA = np.concatenate((times_B_excl_A_array_JJA, timelist_A_JJA), axis=0) # combine B and A time coordinates

    big_list_JJA = times_A_and_B_JJA[:(len(times_B_excl_A_array_JJA)+len(timelist_A_JJA)-1)]
    all_times_JJA = np.sort(big_list_JJA)
    event_timelist_JJA = all_times_JJA
    
else:
    filtered_ds_JJA = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[2]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    event_timelist_JJA = filtered_ds_JJA.time.values 


# In[26]:


# get data set filtered down to only the time steps with interesting IVT over the chosen glacier basin
# the following combines two event lists
if len(basin_set) == 2:

    # get all "A" basin times
    filtered_ds_A_SON = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[3]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_A_SON = filtered_ds_A_SON.time.values

    # get all "B" basin times
    filtered_ds_B_SON = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[3]+"_on_basin_"+str(basin_set[1])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_B_SON = filtered_ds_B_SON.time.values

    # following line says "what does the left set have that the right does not?"
    times_B_excl_A_SON = set(timelist_B_SON) - set(timelist_A_SON) # times in B that are not in A

    times_B_excl_A_array_SON = np.array(list(times_B_excl_A_SON)) # change set into array
    times_A_and_B_SON = np.concatenate((times_B_excl_A_array_SON, timelist_A_SON), axis=0) # combine B and A time coordinates

    big_list_SON = times_A_and_B_SON[:(len(times_B_excl_A_array_SON)+len(timelist_A_SON)-1)]
    all_times_SON = np.sort(big_list_SON)
    event_timelist_SON = all_times_SON

else:
    filtered_ds_SON = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[3]+"_on_basin_"+str(basin_set[0])+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    event_timelist_SON = filtered_ds_SON.time.values 


# In[27]:


def filter_by_time_list(data, time_coords):
    #data = data.sel(time=time_coords)
    data = data.where(data.time.isin(time_coords), drop=True)
    return data


# In[28]:


timelist_DJF = event_timelist_DJF


# In[29]:


timelist_MAM = event_timelist_MAM


# In[30]:


timelist_JJA = event_timelist_JJA


# In[31]:


timelist_SON = event_timelist_SON


# In[32]:


# filter all of the data by the time list

ivt_combined_ds_filtered_all_DJF = filter_by_time_list(ivt_combined_ds_DJF, timelist_DJF)

ivt_ds_filtered_all_DJF = filter_by_time_list(ivt_ds_DJF, timelist_DJF)
da_ivt_filtered_all_DJF = ivt_ds_filtered_all_DJF["Vertical integral of total water vapor flux"]

da_ivt_percentile_filtered_all_DJF = ivt_ds_filtered_all_DJF["Vertical integral of total water vapor flux percentile rank"]
#da_ivt_percentile_filtered_all_DJF *= 100 

ivt_combined_anomaly_filtered_all_DJF = filter_by_time_list(ivt_combined_anomaly_ds_DJF, timelist_DJF)
da_ivt_anomaly_filtered_all_DJF = filter_by_time_list(da_ivt_anomaly_DJF, timelist_DJF)

da_mslp_filtered_all_DJF = filter_by_time_list(da_mslp_DJF, timelist_DJF)

da_mslp_anomaly_filtered_all_DJF = filter_by_time_list(da_mslp_anomaly_DJF, timelist_DJF)


# In[33]:




# In[34]:


# filter all of the data by the time list

ivt_combined_ds_filtered_all_MAM = filter_by_time_list(ivt_combined_ds_MAM, timelist_MAM)

ivt_ds_filtered_all_MAM = filter_by_time_list(ivt_ds_MAM, timelist_MAM)
da_ivt_filtered_all_MAM = ivt_ds_filtered_all_MAM["Vertical integral of total water vapor flux"]

da_ivt_percentile_filtered_all_MAM = ivt_ds_filtered_all_MAM["Vertical integral of total water vapor flux percentile rank"]
#da_ivt_percentile_filtered_all_MAM *= 100 

ivt_combined_anomaly_filtered_all_MAM = filter_by_time_list(ivt_combined_anomaly_ds_MAM, timelist_MAM)
da_ivt_anomaly_filtered_all_MAM = filter_by_time_list(da_ivt_anomaly_MAM, timelist_MAM)

da_mslp_filtered_all_MAM = filter_by_time_list(da_mslp_MAM, timelist_MAM)

da_mslp_anomaly_filtered_all_MAM = filter_by_time_list(da_mslp_anomaly_MAM, timelist_MAM)


# In[35]:


# filter all of the data by the time list

ivt_combined_ds_filtered_all_JJA = filter_by_time_list(ivt_combined_ds_JJA, timelist_JJA)

ivt_ds_filtered_all_JJA = filter_by_time_list(ivt_ds_JJA, timelist_JJA)
da_ivt_filtered_all_JJA = ivt_ds_filtered_all_JJA["Vertical integral of total water vapor flux"]

da_ivt_percentile_filtered_all_JJA = ivt_ds_filtered_all_JJA["Vertical integral of total water vapor flux percentile rank"]
#da_ivt_percentile_filtered_all_JJA *= 100 

ivt_combined_anomaly_filtered_all_JJA = filter_by_time_list(ivt_combined_anomaly_ds_JJA, timelist_JJA)
da_ivt_anomaly_filtered_all_JJA = filter_by_time_list(da_ivt_anomaly_JJA, timelist_JJA)

da_mslp_filtered_all_JJA = filter_by_time_list(da_mslp_JJA, timelist_JJA)

da_mslp_anomaly_filtered_all_JJA = filter_by_time_list(da_mslp_anomaly_JJA, timelist_JJA)


# In[36]:


# filter all of the data by the time list

ivt_combined_ds_filtered_all_SON = filter_by_time_list(ivt_combined_ds_SON, timelist_SON)

ivt_ds_filtered_all_SON = filter_by_time_list(ivt_ds_SON, timelist_SON)
da_ivt_filtered_all_SON = ivt_ds_filtered_all_SON["Vertical integral of total water vapor flux"]

da_ivt_percentile_filtered_all_SON = ivt_ds_filtered_all_SON["Vertical integral of total water vapor flux percentile rank"]
#da_ivt_percentile_filtered_all_SON *= 100 

ivt_combined_anomaly_filtered_all_SON = filter_by_time_list(ivt_combined_anomaly_ds_SON, timelist_SON)
da_ivt_anomaly_filtered_all_SON = filter_by_time_list(da_ivt_anomaly_SON, timelist_SON)

da_mslp_filtered_all_SON = filter_by_time_list(da_mslp_SON, timelist_SON)

da_mslp_anomaly_filtered_all_SON = filter_by_time_list(da_mslp_anomaly_SON, timelist_SON)


# In[37]:


#da_ivt_percentile_filtered_all_DJF = da_ivt_percentile_filtered_all_DJF * 100
#da_ivt_percentile_filtered_all_MAM = da_ivt_percentile_filtered_all_MAM * 100
#da_ivt_percentile_filtered_all_JJA = da_ivt_percentile_filtered_all_JJA * 100
#da_ivt_percentile_filtered_all_SON = da_ivt_percentile_filtered_all_SON * 100


# In[38]:


da_ivt_percentile_filtered_all_DJF = np.multiply(da_ivt_percentile_filtered_all_DJF, 100)
da_ivt_percentile_filtered_all_MAM = np.multiply(da_ivt_percentile_filtered_all_MAM, 100)
da_ivt_percentile_filtered_all_JJA = np.multiply(da_ivt_percentile_filtered_all_JJA, 100)
da_ivt_percentile_filtered_all_SON = np.multiply(da_ivt_percentile_filtered_all_SON, 100)


# In[40]:


# calculate start, end times and duration for each event.

event_start_times = []
event_end_times = []
durations = []

ticks = 0 # start with zero times in the events
for i in range(len(ivt_ds_filtered_all_DJF.time.values)-1):
    
    x = ivt_ds_filtered_all_DJF.isel(time=i).time.values
    y = ivt_ds_filtered_all_DJF.isel(time=i+1).time.values
    
    # if an event-flagged timestamp (x) is more than 24 hours away from the next one (y),
    # count the timestamp x as the end of that event. 3600000000000 = nanoseconds per hour
    
    if (y - x) > (3600000000000 * 24):
        event_end_time = ivt_ds_filtered_all_DJF.isel(time=i).time.values
        event_start_time = ivt_ds_filtered_all_DJF.isel( time=(i-ticks) ).time.values
        
        event_end_times.append(event_end_time+np.timedelta64(3,'h'))
        event_start_times.append(event_start_time)
        
        duration = int( (event_end_time - event_start_time)/3600000000000 ) + 3
        
        durations.append(duration)
        
        ticks = 0 # reset times
        
    else:
        ticks += 1



# In[41]:


indices = range(len(event_start_times))

event_times = xr.Dataset({
    
    'Start time': xr.DataArray(
                data   = event_start_times,
                dims   = ['events'],
                coords = {'events': indices},
                ),
    'End time': xr.DataArray(
            data   = event_end_times,
            dims   = ['events'],
            coords = {'events': indices},
            ),
    'Duration': xr.DataArray(
        data   = durations,  
        dims   = ['events'],
        coords = {'events': indices},
        attrs  = {
            '_FillValue': -999.9,
            'units'     : 'hours'
            }
        ),
    },
)


# In[42]:


if len(basin_set) == 1:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_DJF_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_DJF_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[43]:


# calculate start, end times and duration for each event.

event_start_times = []
event_end_times = []
durations = []

ticks = 0
for i in range(len(ivt_ds_filtered_all_MAM.time.values)-1):
    
    x = ivt_ds_filtered_all_MAM.isel(time=i).time.values
    y = ivt_ds_filtered_all_MAM.isel(time=i+1).time.values
    
    # if an event-flagged timestamp (x) is more than 24 hours away from the next one (y),
    # count the timestamp x as the end of that event. 3600000000000 = nanoseconds per hour
    
    if (y - x) > (3600000000000 * 24):
        event_end_time = ivt_ds_filtered_all_MAM.isel(time=i).time.values
        event_start_time = ivt_ds_filtered_all_MAM.isel( time=(i-ticks) ).time.values
        
        event_end_times.append(event_end_time+np.timedelta64(3,'h'))
        event_start_times.append(event_start_time)
        
        duration = int( (event_end_time - event_start_time)/3600000000000 ) + 3
        
        durations.append(duration)
        
        ticks = 0
        
    else:
        ticks += 1


# In[44]:


indices = range(len(event_start_times))

event_times = xr.Dataset({
    
    'Start time': xr.DataArray(
                data   = event_start_times,
                dims   = ['events'],
                coords = {'events': indices},
                ),
    'End time': xr.DataArray(
            data   = event_end_times,
            dims   = ['events'],
            coords = {'events': indices},
            ),
    'Duration': xr.DataArray(
        data   = durations,  
        dims   = ['events'],
        coords = {'events': indices},
        attrs  = {
            '_FillValue': -999.9,
            'units'     : 'hours'
            }
        ),
    },
)


# In[45]:


if len(basin_set) == 1:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_MAM_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_MAM_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[46]:


# calculate start, end times and duration for each event.

event_start_times = []
event_end_times = []
durations = []

ticks = 0
for i in range(len(ivt_ds_filtered_all_JJA.time.values)-1):
    
    x = ivt_ds_filtered_all_JJA.isel(time=i).time.values
    y = ivt_ds_filtered_all_JJA.isel(time=i+1).time.values
    
    # if an event-flagged timestamp (x) is more than 24 hours away from the next one (y),
    # count the timestamp x as the end of that event. 3600000000000 = nanoseconds per hour
    
    if (y - x) > (3600000000000 * 24):
        event_end_time = ivt_ds_filtered_all_JJA.isel(time=i).time.values
        event_start_time = ivt_ds_filtered_all_JJA.isel( time=(i-ticks) ).time.values
        
        event_end_times.append(event_end_time+np.timedelta64(3,'h'))
        event_start_times.append(event_start_time)
        
        duration = int( (event_end_time - event_start_time)/3600000000000 ) + 3
        
        durations.append(duration)
        
        ticks = 0
        
    else:
        ticks += 1


# In[47]:


indices = range(len(event_start_times))

event_times = xr.Dataset({
    
    'Start time': xr.DataArray(
                data   = event_start_times,
                dims   = ['events'],
                coords = {'events': indices},
                ),
    'End time': xr.DataArray(
            data   = event_end_times,
            dims   = ['events'],
            coords = {'events': indices},
            ),
    'Duration': xr.DataArray(
        data   = durations,  
        dims   = ['events'],
        coords = {'events': indices},
        attrs  = {
            '_FillValue': -999.9,
            'units'     : 'hours'
            }
        ),
    },
)


# In[48]:


if len(basin_set) == 1:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_JJA_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_JJA_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[49]:


# calculate start, end times and duration for each event.

event_start_times = []
event_end_times = []
durations = []

ticks = 0
for i in range(len(ivt_ds_filtered_all_SON.time.values)-1):
    
    x = ivt_ds_filtered_all_SON.isel(time=i).time.values
    y = ivt_ds_filtered_all_SON.isel(time=i+1).time.values
    
    # if an event-flagged timestamp (x) is more than 24 hours away from the next one (y),
    # count the timestamp x as the end of that event. 3600000000000 = nanoseconds per hour
    
    if (y - x) > (3600000000000 * 24):
        event_end_time = ivt_ds_filtered_all_SON.isel(time=i).time.values
        event_start_time = ivt_ds_filtered_all_SON.isel( time=(i-ticks) ).time.values
        
        event_end_times.append(event_end_time+np.timedelta64(3,'h'))
        event_start_times.append(event_start_time)
        
        duration = int( (event_end_time - event_start_time)/3600000000000 ) + 3
        
        durations.append(duration)
        
        ticks = 0
        
    else:
        ticks += 1


# In[50]:


indices = range(len(event_start_times))

event_times = xr.Dataset({
    
    'Start time': xr.DataArray(
                data   = event_start_times,
                dims   = ['events'],
                coords = {'events': indices},
                ),
    'End time': xr.DataArray(
            data   = event_end_times,
            dims   = ['events'],
            coords = {'events': indices},
            ),
    'Duration': xr.DataArray(
        data   = durations,  
        dims   = ['events'],
        coords = {'events': indices},
        attrs  = {
            '_FillValue': -999.9,
            'units'     : 'hours'
            }
        ),
    },
)


# In[51]:


if len(basin_set) == 1:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_SON_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    event_times.to_netcdf(parent_dir+"enhanced_IVT_events_SON_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[52]:


### concatenate ivt_ds_filtered_all_{season} sets together and generate a full data set of events over all times, in order of date.


# In[53]:


# save time coordinates as timearray
if len(basin_set) == 1:
    fname = "event_time_coordinates_basins_DJF_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.txt"
if len(basin_set) == 2:
    fname = "event_time_coordinates_basins_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.txt"
for time in timelist_DJF:
    with open(fname, 'a') as the_file:
        the_file.write(str(time)+'\n')


# In[54]:


# save time coordinates as timearray
if len(basin_set) == 1:
    fname = "event_time_coordinates_basins_MAM_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.txt"
if len(basin_set) == 2:
    fname = "event_time_coordinates_basins_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.txt"
for time in timelist_MAM:
    with open(fname, 'a') as the_file:
        the_file.write(str(time)+'\n')


# In[55]:


# save time coordinates as timearray
if len(basin_set) == 1:
    fname = "event_time_coordinates_basins_JJA_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.txt"
if len(basin_set) == 2:
    fname = "event_time_coordinates_basins_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.txt"
for time in timelist_JJA:
    with open(fname, 'a') as the_file:
        the_file.write(str(time)+'\n')


# In[56]:


# save time coordinates as timearray
if len(basin_set) == 1:
    fname = "event_time_coordinates_basins_SON_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.txt"
if len(basin_set) == 2:
    fname = "event_time_coordinates_basins_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.txt"
for time in timelist_SON:
    with open(fname, 'a') as the_file:
        the_file.write(str(time)+'\n')


# In[57]:


timelist_total = np.concatenate((timelist_DJF, timelist_MAM, timelist_JJA, timelist_SON), axis = 0)
timelist_total = np.sort(timelist_total)


# In[58]:


# save time coordinates as timearray
if len(basin_set) == 1:
    fname = "event_time_coordinates_basins_total_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.txt"
if len(basin_set) == 2:
    fname = "event_time_coordinates_basins_total_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.txt"
for time in timelist_total:
    with open(fname, 'a') as the_file:
        the_file.write(str(time)+'\n')


# In[59]:


if len(basin_set) == 1:
    ds_events_DJF = xr.open_dataset(parent_dir+"enhanced_IVT_events_DJF_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    ds_events_DJF = xr.open_dataset(parent_dir+"enhanced_IVT_events_DJF_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[60]:


if len(basin_set) == 1:
    ds_events_MAM = xr.open_dataset(parent_dir+"enhanced_IVT_events_MAM_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    ds_events_MAM = xr.open_dataset(parent_dir+"enhanced_IVT_events_MAM_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[61]:


if len(basin_set) == 1:
    ds_events_JJA = xr.open_dataset(parent_dir+"enhanced_IVT_events_JJA_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    ds_events_JJA = xr.open_dataset(parent_dir+"enhanced_IVT_events_JJA_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[62]:


if len(basin_set) == 1:
    ds_events_SON = xr.open_dataset(parent_dir+"enhanced_IVT_events_SON_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    ds_events_SON = xr.open_dataset(parent_dir+"enhanced_IVT_events_SON_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[63]:


#ds_combined = xr.concat([ds_events_DJF, ds_events_MAM, ds_events_JJA, ds_events_SON], dim='events')


# In[64]:


#if len(basin_set) == 1:
#    ds_combined.to_netcdf(parent_dir+"enhanced_IVT_events_all_seasons_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

#if len(basin_set) == 2:
#    ds_combined.to_netcdf(parent_dir+"enhanced_IVT_events_all_seasons_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[65]:


date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events


# In[66]:


da_ivt_percentile_filtered_all_DJF = da_ivt_percentile_filtered_all_DJF.sel(time=slice(date1, date2))
ivt_combined_ds_filtered_all_DJF = ivt_combined_ds_filtered_all_DJF.sel(time=slice(date1, date2))
da_ivt_filtered_all_DJF = da_ivt_filtered_all_DJF.sel(time=slice(date1, date2))
da_mslp_filtered_all_DJF = da_mslp_filtered_all_DJF.sel(time=slice(date1, date2))
da_mslp_anomaly_filtered_all_DJF = da_mslp_anomaly_filtered_all_DJF.sel(time=slice(date1, date2))


# In[67]:


DJF_ivt_filtered_data = xr.merge([da_ivt_percentile_filtered_all_DJF, ivt_combined_ds_filtered_all_DJF, da_ivt_filtered_all_DJF])
DJF_ivt_filtered_data.to_netcdf(parent_dir+"IVT_variables_event_only_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_filtered_all_DJF.to_netcdf(parent_dir+"mslp_event_only_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_DJF.to_netcdf(parent_dir+"mslp_anomalies_event_only_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[68]:


da_ivt_percentile_filtered_all_MAM = da_ivt_percentile_filtered_all_MAM.sel(time=slice(date1, date2))
ivt_combined_ds_filtered_all_MAM = ivt_combined_ds_filtered_all_MAM.sel(time=slice(date1, date2))
da_ivt_filtered_all_MAM = da_ivt_filtered_all_MAM.sel(time=slice(date1, date2))
da_mslp_filtered_all_MAM = da_mslp_filtered_all_MAM.sel(time=slice(date1, date2))
da_mslp_anomaly_filtered_all_MAM = da_mslp_anomaly_filtered_all_MAM.sel(time=slice(date1, date2))


# In[69]:


MAM_ivt_filtered_data = xr.merge([da_ivt_percentile_filtered_all_MAM, ivt_combined_ds_filtered_all_MAM, da_ivt_filtered_all_MAM])
MAM_ivt_filtered_data.to_netcdf(parent_dir+"IVT_variables_event_only_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_filtered_all_MAM.to_netcdf(parent_dir+"mslp_event_only_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_MAM.to_netcdf(parent_dir+"mslp_anomalies_event_only_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[70]:


da_ivt_percentile_filtered_all_JJA = da_ivt_percentile_filtered_all_JJA.sel(time=slice(date1, date2))
ivt_combined_ds_filtered_all_JJA = ivt_combined_ds_filtered_all_JJA.sel(time=slice(date1, date2))
da_ivt_filtered_all_JJA = da_ivt_filtered_all_JJA.sel(time=slice(date1, date2))
da_mslp_filtered_all_JJA = da_mslp_filtered_all_JJA.sel(time=slice(date1, date2))
da_mslp_anomaly_filtered_all_JJA = da_mslp_anomaly_filtered_all_JJA.sel(time=slice(date1, date2))


# In[71]:


JJA_ivt_filtered_data = xr.merge([da_ivt_percentile_filtered_all_JJA, ivt_combined_ds_filtered_all_JJA, da_ivt_filtered_all_JJA])
JJA_ivt_filtered_data.to_netcdf(parent_dir+"IVT_variables_event_only_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_filtered_all_JJA.to_netcdf(parent_dir+"mslp_event_only_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_JJA.to_netcdf(parent_dir+"mslp_anomalies_event_only_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[72]:


da_ivt_percentile_filtered_all_SON = da_ivt_percentile_filtered_all_SON.sel(time=slice(date1, date2))
ivt_combined_ds_filtered_all_SON = ivt_combined_ds_filtered_all_SON.sel(time=slice(date1, date2))
da_ivt_filtered_all_SON = da_ivt_filtered_all_SON.sel(time=slice(date1, date2))
da_mslp_filtered_all_SON = da_mslp_filtered_all_SON.sel(time=slice(date1, date2))
da_mslp_anomaly_filtered_all_SON = da_mslp_anomaly_filtered_all_SON.sel(time=slice(date1, date2))


# In[73]:


SON_ivt_filtered_data = xr.merge([da_ivt_percentile_filtered_all_SON, ivt_combined_ds_filtered_all_SON, da_ivt_filtered_all_SON])
SON_ivt_filtered_data.to_netcdf(parent_dir+"IVT_variables_event_only_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_filtered_all_SON.to_netcdf(parent_dir+"mslp_event_only_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_SON.to_netcdf(parent_dir+"mslp_anomalies_event_only_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[ ]:




