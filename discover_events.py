#!/usr/bin/env python
# coding: utf-8

# In[1]:


# discover_events reads in output from calculate_ivt_percentiles, 
# then finds where IVT exceeds a given percentile threshold along all of the given lat/lon points; 
# for now, these points are the bounds of Zwally basins given by a text file.
# The output is a NetCDF file containing IVT magnitude and percentile (w.r.t. 1979-2021 climatology) 
# on each point on the boundary with only the time coordinates saved where the threshold is met. 


# In[2]:


import xarray as xr
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# In[3]:


parent_dir = "/raid01/rendfrey/IVT/"

# In[4]:


basin_indices = [9, 10, 11, 13, 21, 22] # used in get_data_on_glacier to identify glacial basin analyzed

seasons = ["DJF", "MAM", "JJA", "SON"]
season = seasons[0]

percentile_cutoff = 90
threshold = percentile_cutoff / 100.


# In[5]:


ivt_ds = xr.open_dataset(parent_dir+"/total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+season+".nc") # total IVT


# In[6]:


def round_twofive(number):
    x = np.round(number*4)/4
    return x


# In[7]:


def get_data_on_glacier(ds, basin_idx):
    
    # find lat/lon points in a dataset ds which correspond to a certain antarctic glacier basin
    # then select these coordinate pairs in the dataset and return them
    # basin_idx refers to the integer designator of the basin in the text file. see the map in the link below.

    ds_stacked = ds.stack(z=("latitude", "longitude"))
    # read in coordinates within a certain glacier basin
    # coordinates are given in this big long text file
    # see https://earth.gsfc.nasa.gov/cryo/data/polar-altimetry/antarctic-and-greenland-drainage-systems
    # see also Gardner et al., 2018, The Cryosphere.

    basin_coordinate_file = "/raid01/rendfrey/geographic/zwally_basins.txt"
    txt = pd.read_csv(basin_coordinate_file , delim_whitespace=True)
    xds = xr.Dataset.from_dataframe(txt)

    # find basin coordinates and round down if need be (index selection by slice)

    xds_basin = xds.where(xds["Basin"]==basin_idx).dropna(dim='index').isel(index=slice(None, None, 10))
    
    # this script used to just use np.round out of the box, but rounding to the nearest .25 is better.
    # saved the old rounding lines for posterity, however.
    #glacier_lats = np.round(xds_basin["Lat"].values, 2) 
    #glacier_lons = np.round(xds_basin["Lon"].values, 2)

    # round values of the coordinates to nearest .25
    glacier_lats = round_twofive(xds_basin["Lat"].values) 
    glacier_lons = round_twofive(xds_basin["Lon"].values) 

    # create tuples of lat lon pairs found within the glacier
    glacier_lat_lon_tuples = []
    for i in range(0, len(glacier_lats)):
        glacier_lat_lon_tuple = (glacier_lats[i], glacier_lons[i])
        glacier_lat_lon_tuples.append( (glacier_lats[i], glacier_lons[i]) )

    data_lat_lon_tuples = []
    for i in range(0, len(ds_stacked['z'])):
        if ds_stacked['z'].values[i] in glacier_lat_lon_tuples:
            data_lat_lon_tuples.append(ds_stacked['z'].values[i])

    data_on_glacier_list = []
    for i in range(len(data_lat_lon_tuples)):
        data_on_glacier_list.append(ds.sel(latitude=data_lat_lon_tuples[i][0]).sel(longitude=data_lat_lon_tuples[i][1]))

    # zip coordinate pairs together by a new index "p"
    data_on_glacier = xr.concat(data_on_glacier_list, dim="p")
    
    return data_on_glacier


# In[8]:


da_glacier_total = get_data_on_glacier(ivt_ds, 10)


# In[9]:


for basin_index in basin_indices: 

    print(basin_index)
    # get the IVT magnitude along the boundary of the glacial basin
    da_glacier_total = get_data_on_glacier(ivt_ds, basin_index)
    
    # calculate the median along the boundary of each timestep
    da_glacier_total_median = da_glacier_total.median(dim="p")

    # filter dataset by percentile threshold, drop times that do not meet IVT threshold
    da_glacier_total_filtered_dropped = da_glacier_total.where(da_glacier_total_median>float(threshold)).dropna(dim="time")

    # save filtered dataset as its own file
    da_glacier_total_filtered_dropped.to_netcdf(parent_dir+"vertical_integral_of_total_wv_flux_"+season+"_on_basin_"+str(basin_index)+"_greater_or_equal_to_the_"+str(percentile_cutoff)+"th_percentile.nc")
    print(da_glacier_total_filtered_dropped)


# In[ ]:




