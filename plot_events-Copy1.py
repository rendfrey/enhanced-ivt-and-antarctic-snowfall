#!/usr/bin/env python
# coding: utf-8

# In[31]:


import math
import xarray as xr
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


# In[32]:


#parent_dir = "/nfs/turbo/clasp-aepayne/ERA5/rendfrey/southern_hemisphere/"
parent_dir = "/raid01/rendfrey/IVT_and_MSLP/event_only/"
ivt_dir = "/raid01/rendfrey/IVT/"

# In[33]:


basin = "amundsen" # amundsen or amery

seasons = ["DJF", "MAM", "JJA", "SON"]

percentile_cutoff = 90
threshold = percentile_cutoff / 100.


# In[4]:


#ds_ars = xr.open_mfdataset("/nfs/turbo/clasp-aepayne/ERA5/rendfrey/southern_hemisphere/AR_tags/cs_era/*")


# In[5]:

ds_events_DJF = xr.open_dataset("/raid01/rendfrey/event_lists/enhanced_IVT_events_DJF_basins_21_22_95th_percentile_IVT.nc")
ds_events_MAM = xr.open_dataset("/raid01/rendfrey/event_lists/enhanced_IVT_events_MAM_basins_21_22_95th_percentile_IVT.nc")
ds_events_JJA = xr.open_dataset("/raid01/rendfrey/event_lists/enhanced_IVT_events_JJA_basins_21_22_95th_percentile_IVT.nc")
ds_events_SON = xr.open_dataset("/raid01/rendfrey/event_lists/enhanced_IVT_events_SON_basins_21_22_95th_percentile_IVT.nc")

ds_northward_ivt = xr.open_mfdataset("/raid01/rendfrey/IVT_and_MSLP/vertical_integral_of_northward_water_vapour_flux/*")
ds_eastward_ivt = xr.open_mfdataset("/raid01/rendfrey/IVT_and_MSLP/vertical_integral_of_eastward_water_vapour_flux/*")

# open IVT, MSLP files

# open IVT files
ivt_ds_DJF_magnitude = xr.open_dataset(ivt_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[0]+".nc") # total IVT calculated by a previous script
ivt_ds_MAM_magnitude = xr.open_dataset(ivt_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[1]+".nc") # total IVT calculated by a previous script
ivt_ds_JJA_magnitude = xr.open_dataset(ivt_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[2]+".nc") # total IVT calculated by a previous script
ivt_ds_SON_magnitude = xr.open_dataset(ivt_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+seasons[3]+".nc") # total IVT calculated by a previous script

DJF_event_ivt_sets = []
DJF_event_northward_ivt_sets = []
DJF_event_eastward_ivt_sets = []
for i in range(len(ds_events_DJF["Start time"])):
	ds_event = ivt_ds_DJF_magnitude.sel(time=slice(ds_events_DJF["Start time"].isel(events=i).values, ds_events_DJF["End time"].isel(events=i).values))
	northward_ivt = ds_northward_ivt.sel(time=slice(ds_events_DJF["Start time"].isel(events=i).values, ds_events_DJF["End time"].isel(events=i).values))
	eastward_ivt = ds_eastward_ivt.sel(time=slice(ds_events_DJF["Start time"].isel(events=i).values, ds_events_DJF["End time"].isel(events=i).values))

	DJF_event_ivt_sets.append(ds_event)
	DJF_event_northward_ivt_sets.append(northward_ivt)
	DJF_event_eastward_ivt_sets.append(eastward_ivt)

ds_ivt_DJF_magnitudes = xr.concat(DJF_event_ivt_sets, dim="time")
ds_ivt_DJF_north = xr.concat(DJF_event_northward_ivt_sets, dim="time")
ds_ivt_DJF_east = xr.concat(DJF_event_eastward_ivt_sets, dim="time")

ds_ivt_DJF = xr.merge([ds_ivt_DJF_magnitudes, ds_ivt_DJF_north, ds_ivt_DJF_east])

MAM_event_ivt_sets = []
MAM_event_northward_ivt_sets = []
MAM_event_eastward_ivt_sets = []
for i in range(len(ds_events_MAM["Start time"])):
        ds_event = ivt_ds_MAM_magnitude.sel(time=slice(ds_events_MAM["Start time"].isel(events=i).values, ds_events_MAM["End time"].isel(events=i).values))
        northward_ivt = ds_northward_ivt.sel(time=slice(ds_events_MAM["Start time"].isel(events=i).values, ds_events_MAM["End time"].isel(events=i).values))
        eastward_ivt = ds_eastward_ivt.sel(time=slice(ds_events_MAM["Start time"].isel(events=i).values, ds_events_MAM["End time"].isel(events=i).values))

        MAM_event_ivt_sets.append(ds_event)
        MAM_event_northward_ivt_sets.append(northward_ivt)
        MAM_event_eastward_ivt_sets.append(eastward_ivt)

ds_ivt_MAM_magnitudes = xr.concat(MAM_event_ivt_sets, dim="time")
ds_ivt_MAM_north = xr.concat(MAM_event_northward_ivt_sets, dim="time")
ds_ivt_MAM_east = xr.concat(MAM_event_eastward_ivt_sets, dim="time")

ds_ivt_MAM = xr.merge([ds_ivt_MAM_magnitudes, ds_ivt_MAM_north, ds_ivt_MAM_east])

JJA_event_ivt_sets = []
JJA_event_northward_ivt_sets = []
JJA_event_eastward_ivt_sets = []
for i in range(len(ds_events_JJA["Start time"])):
        ds_event = ivt_ds_JJA_magnitude.sel(time=slice(ds_events_JJA["Start time"].isel(events=i).values, ds_events_JJA["End time"].isel(events=i).values))
        northward_ivt = ds_northward_ivt.sel(time=slice(ds_events_JJA["Start time"].isel(events=i).values, ds_events_JJA["End time"].isel(events=i).values))
        eastward_ivt = ds_eastward_ivt.sel(time=slice(ds_events_JJA["Start time"].isel(events=i).values, ds_events_JJA["End time"].isel(events=i).values))

        JJA_event_ivt_sets.append(ds_event)
        JJA_event_northward_ivt_sets.append(northward_ivt)
        JJA_event_eastward_ivt_sets.append(eastward_ivt)

ds_ivt_JJA_magnitudes = xr.concat(JJA_event_ivt_sets, dim="time")
ds_ivt_JJA_north = xr.concat(JJA_event_northward_ivt_sets, dim="time")
ds_ivt_JJA_east = xr.concat(JJA_event_eastward_ivt_sets, dim="time")

ds_ivt_JJA = xr.merge([ds_ivt_JJA_magnitudes, ds_ivt_JJA_north, ds_ivt_JJA_east])

JJA_event_ivt_sets = []
JJA_event_northward_ivt_sets = []
JJA_event_eastward_ivt_sets = []
for i in range(len(ds_events_JJA["Start time"])):
        ds_event = ivt_ds_JJA_magnitude.sel(time=slice(ds_events_JJA["Start time"].isel(events=i).values, ds_events_JJA["End time"].isel(events=i).values))
        northward_ivt = ds_northward_ivt.sel(time=slice(ds_events_JJA["Start time"].isel(events=i).values, ds_events_JJA["End time"].isel(events=i).values))
        eastward_ivt = ds_eastward_ivt.sel(time=slice(ds_events_JJA["Start time"].isel(events=i).values, ds_events_JJA["End time"].isel(events=i).values))

        JJA_event_ivt_sets.append(ds_event)
        JJA_event_northward_ivt_sets.append(northward_ivt)
        JJA_event_eastward_ivt_sets.append(eastward_ivt)

ds_ivt_JJA_magnitudes = xr.concat(JJA_event_ivt_sets, dim="time")
ds_ivt_JJA_north = xr.concat(JJA_event_northward_ivt_sets, dim="time")
ds_ivt_JJA_east = xr.concat(JJA_event_eastward_ivt_sets, dim="time")

ds_ivt_JJA = xr.merge([ds_ivt_JJA_magnitudes, ds_ivt_JJA_north, ds_ivt_JJA_east])


SON_event_ivt_sets = []
SON_event_northward_ivt_sets = []
SON_event_eastward_ivt_sets = []
for i in range(len(ds_events_SON["Start time"])):
        ds_event = ivt_ds_SON_magnitude.sel(time=slice(ds_events_SON["Start time"].isel(events=i).values, ds_events_SON["End time"].isel(events=i).values))
        northward_ivt = ds_northward_ivt.sel(time=slice(ds_events_SON["Start time"].isel(events=i).values, ds_events_SON["End time"].isel(events=i).values))
        eastward_ivt = ds_eastward_ivt.sel(time=slice(ds_events_SON["Start time"].isel(events=i).values, ds_events_SON["End time"].isel(events=i).values))

        SON_event_ivt_sets.append(ds_event)
        SON_event_northward_ivt_sets.append(northward_ivt)
        SON_event_eastward_ivt_sets.append(eastward_ivt)

ds_ivt_SON_magnitudes = xr.concat(SON_event_ivt_sets, dim="time")
ds_ivt_SON_north = xr.concat(SON_event_northward_ivt_sets, dim="time")
ds_ivt_SON_east = xr.concat(SON_event_eastward_ivt_sets, dim="time")

ds_ivt_SON = xr.merge([ds_ivt_SON_magnitudes, ds_ivt_SON_north, ds_ivt_SON_east])

#quit()

if basin == "amundsen":
    basin_set = [21, 22]
elif basin == "amery":
    basin_set = [9, 11]
#basin_set = [10]

seasons = ["DJF", "MAM", "JJA", "SON"]

percentile_cutoff = 95
threshold = percentile_cutoff / 100.


# In[36]:


#ds_ivt_DJF = xr.open_dataset(parent_dir+"IVT_variables_event_only_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
ds_mslp_filtered_all_DJF = xr.open_dataset(parent_dir+"mslp_event_only_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
#ds_mslp_all_DJF = xr.open_mfdataset(parent_dir+"/mslp0709/*.nc")
#ds_mslp_all_DJF = ds_mslp_all_DJF.sel(time=(ds_mslp_all_DJF['time.season']=="DJF"))
ds_mslp_anomaly_filtered_all_DJF = xr.open_dataset(parent_dir+"mslp_anomalies_event_only_DJF_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_DJF = ds_mslp_anomaly_filtered_all_DJF["msl"]


# In[37]:


#ds_ivt_MAM = xr.open_dataset(parent_dir+"IVT_variables_event_only_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
ds_mslp_filtered_all_MAM = xr.open_dataset(parent_dir+"mslp_event_only_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
#ds_mslp_all_MAM = xr.open_mfdataset(parent_dir+"/mslp0709/*.nc")
#ds_mslp_all_MAM = ds_mslp_all_MAM.sel(time=(ds_mslp_all_MAM['time.season']=="MAM"))
ds_mslp_anomaly_filtered_all_MAM = xr.open_dataset(parent_dir+"mslp_anomalies_event_only_MAM_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_MAM = ds_mslp_anomaly_filtered_all_MAM["msl"]


# In[38]:


#ds_ivt_JJA = xr.open_dataset(parent_dir+"IVT_variables_event_only_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
ds_mslp_filtered_all_JJA = xr.open_dataset(parent_dir+"mslp_event_only_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
#ds_mslp_all_JJA = xr.open_mfdataset(parent_dir+"/mslp0709/*.nc")
#ds_mslp_all_JJA = ds_mslp_all_JJA.sel(time=(ds_mslp_all_JJA['time.season']=="JJA"))
ds_mslp_anomaly_filtered_all_JJA = xr.open_dataset(parent_dir+"mslp_anomalies_event_only_JJA_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_JJA = ds_mslp_anomaly_filtered_all_JJA["msl"]


# In[39]:


#ds_ivt_SON = xr.open_dataset(parent_dir+"IVT_variables_event_only_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
ds_mslp_filtered_all_SON = xr.open_dataset(parent_dir+"mslp_event_only_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
#ds_mslp_all_SON = xr.open_mfdataset(parent_dir+"/mslp0709/*.nc")
#ds_mslp_all_SON = ds_mslp_all_SON.sel(time=(ds_mslp_all_SON['time.season']=="SON"))
ds_mslp_anomaly_filtered_all_SON = xr.open_dataset(parent_dir+"mslp_anomalies_event_only_SON_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")
da_mslp_anomaly_filtered_all_SON = ds_mslp_anomaly_filtered_all_SON["msl"]


# In[ ]:


da_ivt_percentile_filtered_all_DJF = ds_ivt_DJF["Vertical integral of total water vapor flux percentile rank"]
ivt_component_one_DJF = ds_ivt_DJF["p71.162"]
ivt_component_two_DJF = ds_ivt_DJF["p72.162"]
ivt_combined_ds_filtered_all_DJF = xr.merge([ivt_component_one_DJF, ivt_component_two_DJF])
da_ivt_filtered_all_DJF = ds_ivt_DJF["Vertical integral of total water vapor flux"]


# In[ ]:


da_ivt_percentile_filtered_all_MAM = ds_ivt_MAM["Vertical integral of total water vapor flux percentile rank"]
ivt_component_one_MAM = ds_ivt_MAM["p71.162"]
ivt_component_two_MAM = ds_ivt_MAM["p72.162"]
ivt_combined_ds_filtered_all_MAM = xr.merge([ivt_component_one_MAM, ivt_component_two_MAM])
da_ivt_filtered_all_MAM = ds_ivt_MAM["Vertical integral of total water vapor flux"]


# In[ ]:


da_ivt_percentile_filtered_all_JJA = ds_ivt_JJA["Vertical integral of total water vapor flux percentile rank"]
ivt_component_one_JJA = ds_ivt_JJA["p71.162"]
ivt_component_two_JJA = ds_ivt_JJA["p72.162"]
ivt_combined_ds_filtered_all_JJA = xr.merge([ivt_component_one_JJA, ivt_component_two_JJA])
da_ivt_filtered_all_JJA = ds_ivt_JJA["Vertical integral of total water vapor flux"]


# In[ ]:


da_ivt_percentile_filtered_all_SON = ds_ivt_SON["Vertical integral of total water vapor flux percentile rank"]
ivt_component_one_SON = ds_ivt_SON["p71.162"]
ivt_component_two_SON = ds_ivt_SON["p72.162"]
ivt_combined_ds_filtered_all_SON = xr.merge([ivt_component_one_SON, ivt_component_two_SON])
da_ivt_filtered_all_SON = ds_ivt_SON["Vertical integral of total water vapor flux"]


# In[ ]:


### IVT plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events

ivt_plot_color_var_DJF = da_ivt_percentile_filtered_all_DJF.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_color_var_DJF.attrs["units"] = "Percentile rank of IVT"
ivt_plot_color_min_DJF = 0
ivt_plot_color_max_DJF = 100
ivt_plot_extents_DJF = [-180, 180, -90, -65]
ivt_plot_colorscheme_DJF = 'GnBu'

ivt_plot_quiver_var_DJF = ivt_combined_ds_filtered_all_DJF.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_quiver_scale_DJF = 1700

ivt_plot_contour_var_DJF = da_ivt_filtered_all_DJF.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_contour_var_DJF.attrs["units"] = "kg m^-1 s^-1"
ivt_plot_contour_min_DJF = 0
ivt_plot_contour_max_DJF = 200
ivt_plot_contour_levels_DJF = 50


# In[ ]:


ivt_plot_color_var_MAM = da_ivt_percentile_filtered_all_MAM.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_color_var_MAM.attrs["units"] = "Percentile rank of IVT"
ivt_plot_color_min_MAM = 0
ivt_plot_color_max_MAM = 100
ivt_plot_extents_MAM = [-180, 180, -90, -65]
ivt_plot_colorscheme_MAM = 'GnBu'

ivt_plot_quiver_var_MAM = ivt_combined_ds_filtered_all_MAM.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_quiver_scale_MAM = 1700

ivt_plot_contour_var_MAM = da_ivt_filtered_all_MAM.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_contour_var_MAM.attrs["units"] = "kg m^-1 s^-1"
ivt_plot_contour_min_MAM = 0
ivt_plot_contour_max_MAM = 200
ivt_plot_contour_levels_MAM = 50


# In[ ]:


ivt_plot_color_var_JJA = da_ivt_percentile_filtered_all_JJA.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_color_var_JJA.attrs["units"] = "Percentile rank of IVT"
ivt_plot_color_min_JJA = 0
ivt_plot_color_max_JJA = 100
ivt_plot_extents_JJA = [-180, 180, -90, -65]
ivt_plot_colorscheme_JJA = 'GnBu'

ivt_plot_quiver_var_JJA = ivt_combined_ds_filtered_all_JJA.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_quiver_scale_JJA = 1700

ivt_plot_contour_var_JJA = da_ivt_filtered_all_JJA.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_contour_var_JJA.attrs["units"] = "kg m^-1 s^-1"
ivt_plot_contour_min_JJA = 0
ivt_plot_contour_max_JJA = 200
ivt_plot_contour_levels_JJA = 50


# In[ ]:


ivt_plot_color_var_SON = da_ivt_percentile_filtered_all_SON.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_color_var_SON.attrs["units"] = "Percentile rank of IVT"
ivt_plot_color_min_SON = 0
ivt_plot_color_max_SON = 100
ivt_plot_extents_SON = [-180, 180, -90, -65]
ivt_plot_colorscheme_SON = 'GnBu'

ivt_plot_quiver_var_SON = ivt_combined_ds_filtered_all_SON.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_quiver_scale_SON = 1700

ivt_plot_contour_var_SON = da_ivt_filtered_all_SON.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_contour_var_SON.attrs["units"] = "kg $\mathregular{m^{-1}}$ $\mathregular{s^{-1}}$"
ivt_plot_contour_min_SON = 0
ivt_plot_contour_max_SON = 200
ivt_plot_contour_levels_SON = 50

ivt_plot_color_var_DJF *= 100
ivt_plot_color_var_MAM *= 100
ivt_plot_color_var_JJA *= 100
ivt_plot_color_var_SON *= 100
# In[ ]:


def round_twofive(number):
    x = np.round(number*4)/4
    return x


# In[ ]:


def get_basin_bounds(basin_idx):

    # find lat/lon points in a dataset ds which correspond to a certain antarctic glacier basin
    # then select these coordinate pairs in the dataset and return them
    # basin_idx refers to the integer designator of the basin in the text file. see the map in the link below.
    ds = ivt_plot_color_var_SON # this choice of data set doesn't matter as long as it contains the right geographic region 
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
    # this used to just use np.round out of the box, but rounding to the nearest .25 is better.
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


# In[ ]:


bound1 = get_basin_bounds(9)
bound2 = get_basin_bounds(11)
bound3 = get_basin_bounds(21)
bound4 = get_basin_bounds(22)


# In[ ]:

elevation_ds = xr.open_dataset("/raid01/rendfrey/geographic/BedMachineAntarctica-v3.nc")
elev_ds = elevation_ds.isel(x=slice(None, None, 100), y=slice(None, None, 100))
subset = elevation_ds.isel(x=slice(None, None, 100), y=slice(None, None, 100))
data = xr.open_dataset("/raid01/rendfrey/geographic/BedMachineAntarctica-v3.nc", 
                         chunks={'x': 5000, 'y': 5000}).sortby('y')
data["surface"] /= 1000
data.surface.attrs["units"] = "kilometers"


# In[ ]:


from numpy import s_
fig = plt.figure(figsize=(10, 10), dpi=300)
ax = plt.axes(projection=ccrs.SouthPolarStereo())
ax.coastlines()

gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl.xlabel_style = {'size': 20}
gl.ylabel_style = {'size': 20}

gl.ylocator = mticker.FixedLocator([-80, -70])
gl.xlocator = mticker.FixedLocator([-120, -60, -90, 90, 60, 0, 120, 180])

gl.rotate_labels = False

gl2 = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)

gl2.ylocator = mticker.FixedLocator([-80, -70])
gl2.xlocator = mticker.FixedLocator([-120, -60, -90, 90, 60, 0, 120, 180])

pl = data.surface.isel(x=s_[0::50], y=s_[0::50]).plot.contourf(levels = 10, cmap="Greys", alpha = 0.7, add_colorbar=False)

plt.scatter(bound1["longitude"], bound1["latitude"],
     color='black', marker='o', s=1,
     transform=ccrs.PlateCarree(), zorder=21
     )

plt.scatter(bound2["longitude"], bound2["latitude"],
     color='black', marker='o', s=1,
     transform=ccrs.PlateCarree(), zorder=22
     )

plt.scatter(bound3["longitude"], bound3["latitude"],
     color='black', marker='o', s=1,
     transform=ccrs.PlateCarree(), zorder=23
     )

plt.scatter(bound4["longitude"], bound4["latitude"],
     color='black', marker='o', s=1,
     transform=ccrs.PlateCarree(), zorder=24
     )
    
tick_labels = [0, 1, 2, 3, 4]
#fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.21, 0.03, 0.59, 0.04])
#fig.colorbar(pl_DJF, cax=cbar_ax, orientation="horizontal")
cbar_ax.tick_params(labelsize=20)
fig.colorbar(pl, orientation="horizontal", cax=cbar_ax, extend='both', ticks=tick_labels).set_label(label='Surface height (km)',size=20,weight='bold')
ax.set_extent([-180, 180, -90, -65], 
          ccrs.PlateCarree())

plt.savefig("basin_bounds_and_topography.png")

plt.show()


# In[ ]:


if basin == "amery":
    glacier_bound_1 = bound1
    glacier_bound_2 = bound2

elif basin == "amundsen":
    glacier_bound_1 = bound3
    glacier_bound_2 = bound4
    
# bound1 = glacier basin 9, bound2 = g.b. 11, bound3 = g.b. 21, bound4 = g.b. 22


# In[ ]:


# IVT plot
num_cols = 4
num_rows = 1
figure_size = (20, 15)
proj_type = ccrs.SouthPolarStereo()

# defining the figure
fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols, figsize=figure_size, dpi=300,
                        subplot_kw=dict(projection=proj_type, facecolor='w') )
#######################################################################

# COLOR PLOT
pl_DJF = ivt_plot_color_var_DJF.plot(transform=ccrs.PlateCarree(), 
                                   vmin=ivt_plot_color_min_DJF, vmax=ivt_plot_color_max_DJF,
                                   add_colorbar=False, ax = axs[0])
pl_DJF.axes.set_extent(ivt_plot_extents_DJF, 
              ccrs.PlateCarree())
pl_DJF.set_cmap(ivt_plot_colorscheme_DJF)
pl_DJF.axes.coastlines(linewidth = 2.0)

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample_DJF = ivt_plot_quiver_var_DJF.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv_DJF = resample_DJF.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="black",
                            transform=ccrs.PlateCarree(), add_guide = False,
                            scale = ivt_plot_quiver_scale_DJF, width = 0.004, ax = axs[0]
                            )

# CONTOUR PLOT
pl_levels_DJF = list(range(ivt_plot_contour_min_DJF, ivt_plot_contour_max_DJF, ivt_plot_contour_levels_DJF))

pl_contours_DJF = pl_DJF.axes.contour(ivt_plot_contour_var_DJF['longitude'], ivt_plot_contour_var_DJF['latitude'], 
                              ivt_plot_contour_var_DJF, 
                              colors=['gray'], levels = pl_levels_DJF,
                              transform=ccrs.PlateCarree())

#pl_contours_labels_DJF = pl.axes.clabel(pl_contours, inline=True, fontsize=16)

# MAM
# COLOR PLOT
pl_MAM = ivt_plot_color_var_MAM.plot(transform=ccrs.PlateCarree(), 
                                   vmin=ivt_plot_color_min_MAM, vmax=ivt_plot_color_max_MAM,
                                   add_colorbar=False, ax = axs[1])
pl_MAM.axes.set_extent(ivt_plot_extents_MAM, 
              ccrs.PlateCarree())
pl_MAM.set_cmap(ivt_plot_colorscheme_MAM)
pl_MAM.axes.coastlines(linewidth = 2.0)

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample_MAM = ivt_plot_quiver_var_MAM.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv_MAM = resample_MAM.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="black",
                            transform=ccrs.PlateCarree(), add_guide = False,
                            scale = ivt_plot_quiver_scale_MAM, width = 0.004, ax = axs[1]
                            )

# CONTOUR PLOT
pl_levels_MAM = list(range(ivt_plot_contour_min_MAM, ivt_plot_contour_max_MAM, ivt_plot_contour_levels_MAM))

pl_contours_MAM = pl_MAM.axes.contour(ivt_plot_contour_var_MAM['longitude'], ivt_plot_contour_var_MAM['latitude'], 
                              ivt_plot_contour_var_MAM, 
                              colors=['gray'], levels = pl_levels_MAM,
                              transform=ccrs.PlateCarree())

#pl_contours_labels_MAM = pl.axes.clabel(pl_contours, inline=True, fontsize=16)

# JJA
# COLOR PLOT
pl_JJA = ivt_plot_color_var_JJA.plot(transform=ccrs.PlateCarree(), 
                                   vmin=ivt_plot_color_min_JJA, vmax=ivt_plot_color_max_JJA,
                                   add_colorbar=False, ax = axs[2])
pl_JJA.axes.set_extent(ivt_plot_extents_JJA, 
              ccrs.PlateCarree())
pl_JJA.set_cmap(ivt_plot_colorscheme_JJA)
pl_JJA.axes.coastlines(linewidth = 2.0)

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample_JJA = ivt_plot_quiver_var_JJA.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv_JJA = resample_JJA.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="black",
                            transform=ccrs.PlateCarree(), add_guide = False,
                            scale = ivt_plot_quiver_scale_JJA, width = 0.004, ax = axs[2]
                            )

# CONTOUR PLOT
pl_levels_JJA = list(range(ivt_plot_contour_min_JJA, ivt_plot_contour_max_JJA, ivt_plot_contour_levels_JJA))

pl_contours_JJA = pl_JJA.axes.contour(ivt_plot_contour_var_JJA['longitude'], ivt_plot_contour_var_JJA['latitude'], 
                              ivt_plot_contour_var_JJA, 
                              colors=['gray'], levels = pl_levels_JJA,
                              transform=ccrs.PlateCarree())

#pl_contours_labels_JJA = pl.axes.clabel(pl_contours, inline=True, fontsize=16)

# SON
# COLOR PLOT
pl_SON = ivt_plot_color_var_SON.plot(transform=ccrs.PlateCarree(), 
                                   vmin=ivt_plot_color_min_SON, vmax=ivt_plot_color_max_SON,
                                   add_colorbar=False, ax = axs[3])
pl_SON.axes.set_extent(ivt_plot_extents_SON, 
              ccrs.PlateCarree())
pl_SON.set_cmap(ivt_plot_colorscheme_SON)
pl_SON.axes.coastlines(linewidth = 2.0)

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample_SON = ivt_plot_quiver_var_SON.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv_SON = resample_SON.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="black",
                            transform=ccrs.PlateCarree(),
                            scale = ivt_plot_quiver_scale_SON, width = 0.004, ax = axs[3]
                            )
plt.quiverkey(quiv_SON, 0.1, -0.05, 
              ivt_plot_contour_max_SON, r' '+str(ivt_plot_contour_max_SON)+' '+ivt_plot_contour_var_SON.attrs["units"], 
              labelpos='S', fontproperties={'weight': 'bold', 'size': '14'})

# CONTOUR PLOT
pl_levels_SON = list(range(ivt_plot_contour_min_SON, ivt_plot_contour_max_SON, ivt_plot_contour_levels_SON))

pl_contours_SON = pl_SON.axes.contour(ivt_plot_contour_var_SON['longitude'], ivt_plot_contour_var_SON['latitude'], 
                              ivt_plot_contour_var_SON, 
                              colors=['gray'], levels = pl_levels_SON,
                              transform=ccrs.PlateCarree())

#pl_contours_labels_SON = pl.axes.clabel(pl_contours, inline=True, fontsize=16)


# general plotting settings + extra lines
"""
axs[0].set_title("(a) DJF", fontsize=24, weight='bold')
axs[1].set_title("(b) MAM", fontsize=24, weight='bold')
axs[2].set_title("(c) JJA", fontsize=24, weight='bold')
axs[3].set_title("(d) SON", fontsize=24, weight='bold')

# plot basin bounds
pl_bound_1 = axs[0].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_2 = axs[0].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

pl_bound_11 = axs[1].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_21 = axs[1].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

pl_bound_12 = axs[2].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_22 = axs[2].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

pl_bound_13 = axs[3].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_23 = axs[3].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )


gl1 = axs[0].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl2 = axs[1].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl3 = axs[2].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl4 = axs[3].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)

gl1.xlabel_style = {'size': 20}
gl1.ylabel_style = {'size': 20}
gl1.xlabels_top = False
gl1.ylabels_right = False

gl2.xlabel_style = {'size': 20}
gl2.ylabel_style = {'size': 20}
gl2.xlabels_top = False
gl2.ylabels_left = False
gl2.ylabels_right = False

gl3.xlabel_style = {'size': 20}
gl3.ylabel_style = {'size': 20}
gl3.xlabels_top = False
gl3.ylabels_left = False
gl3.ylabels_right = False

gl4.xlabel_style = {'size': 20}
gl4.ylabel_style = {'size': 20}
gl4.xlabels_top = False
gl4.ylabels_left = False
gl1.rotate_labels = False
gl2.rotate_labels = False
gl3.rotate_labels = False
gl4.rotate_labels = False

gl1.ylocator = mticker.FixedLocator([-80, -70, -60])
gl1.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl2.ylocator = mticker.FixedLocator([-80, -70, -60])
gl2.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl3.ylocator = mticker.FixedLocator([-80, -70, -60])
gl3.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl4.ylocator = mticker.FixedLocator([-80, -70, -60])
gl4.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])
"""
#fig.subplots_adjust(bottom=0.25)
cbar_ax = fig.add_axes([0.20, 0.35, 0.60, 0.02])
#fig.colorbar(pl_DJF, cax=cbar_ax, orientation="horizontal")
cbar_ax.tick_params(labelsize=24)

fig.colorbar(pl_DJF, orientation="horizontal", cax=cbar_ax).set_label(label='Percentile of IVT',size=24,weight='bold')
fig.subplots_adjust(wspace=0.01,hspace=0.01)



#axs.set_title('Water vapor transport percentiles')

"""
if len(basin_set) == 1:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basin_"+season+"_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")
if len(basin_set) == 2:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basins_"+season+"_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")
"""

if basin == "amery":
    plt.savefig(parent_dir+"amery_ivt_event_composites.png")
elif basin == "amundsen":
    plt.savefig(parent_dir+"amundsen_ivt_event_composites.png")

plt.show()
#plt.close()


# In[54]:


# MSLP plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events
#mslp_plot_color_var_DJF = ds_mslp_all_DJF["msl"].sel(time=slice(date1, date2)).mean(dim="time")
#mslp_plot_color_var_DJF = mslp_plot_color_var_DJF / 100

mslp_plot_color_var_DJF = da_mslp_anomaly_filtered_all_DJF.sel(time=slice(date1, date2)).mean(dim="time")
mslp_plot_color_var_DJF.attrs["units"] = "hPa"
mslp_plot_contour_var_DJF = da_mslp_anomaly_filtered_all_DJF.sel(time=slice(date1, date2)).mean(dim="time")

###


# In[46]:


# MSLP plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events

mslp_plot_color_var_MAM = ds_mslp_filtered_all_MAM["msl"].sel(time=slice(date1, date2)).mean(dim="time")

#mslp_plot_color_var_MAM = ds_mslp_all_MAM["msl"].sel(time=slice(date1, date2)).mean(dim="time")
#mslp_plot_color_var_MAM = mslp_plot_color_var_MAM / 100

mslp_plot_color_var_MAM = da_mslp_anomaly_filtered_all_MAM.sel(time=slice(date1, date2)).mean(dim="time")
mslp_plot_color_var_MAM.attrs["units"] = "hPa"
mslp_plot_contour_var_MAM = da_mslp_anomaly_filtered_all_MAM.sel(time=slice(date1, date2)).mean(dim="time")

###


# In[47]:


# MSLP plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events
#mslp_plot_color_var_JJA = ds_mslp_all_JJA["msl"].sel(time=slice(date1, date2)).mean(dim="time")
#mslp_plot_color_var_JJA = mslp_plot_color_var_JJA / 100

mslp_plot_color_var_JJA = da_mslp_anomaly_filtered_all_JJA.sel(time=slice(date1, date2)).mean(dim="time")
mslp_plot_color_var_JJA.attrs["units"] = "hPa"
mslp_plot_contour_var_JJA = da_mslp_anomaly_filtered_all_JJA.sel(time=slice(date1, date2)).mean(dim="time")

###


# In[52]:


# MSLP plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events
#mslp_plot_color_var_SON = ds_mslp_all_SON["msl"].sel(time=slice(date1, date2)).mean(dim="time")
#mslp_plot_color_var_SON = mslp_plot_color_var_SON / 100

mslp_plot_color_var_SON = da_mslp_anomaly_filtered_all_SON.sel(time=slice(date1, date2)).mean(dim="time")
mslp_plot_color_var_SON.attrs["units"] = "hPa"
mslp_plot_contour_var_SON = da_mslp_anomaly_filtered_all_SON.sel(time=slice(date1, date2)).mean(dim="time")

#mslp_plot_color_min = -10
#mslp_plot_color_max = 10
mslp_plot_color_min = 980
mslp_plot_color_max = 1020
mslp_plot_extents = [-180, 180, -90, -65]
#mslp_plot_colorscheme = 'bwr'
mslp_plot_colorscheme = 'GnBu_r'
mslp_plot_contour_min = -20
mslp_plot_contour_max = 20
mslp_plot_contour_levels = 10
###


# In[53]:


# MSLP plot
num_cols = 4
num_rows = 1
figure_size = (20, 20)
proj_type = ccrs.SouthPolarStereo()

# defining the figure
fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols, figsize=figure_size, dpi=300,
                        subplot_kw=dict(projection=proj_type, facecolor='w') )
# COLOR PLOT
pl_DJF = mslp_plot_color_var_DJF.plot(transform=ccrs.PlateCarree(), 
                                   vmin=mslp_plot_color_min, vmax=mslp_plot_color_max,
                                     add_colorbar=False, ax = axs[0])
pl_DJF.axes.set_extent(mslp_plot_extents, 
              ccrs.PlateCarree())
pl_DJF.set_cmap(mslp_plot_colorscheme)
pl_DJF.axes.coastlines(linewidth = 2.0)

# CONTOUR PLOT
pl_DJF_levels = list(range(mslp_plot_contour_min, mslp_plot_contour_max, mslp_plot_contour_levels))

pl_DJF_contours = pl_DJF.axes.contour(mslp_plot_contour_var_DJF['longitude'], mslp_plot_contour_var_DJF['latitude'], 
                              mslp_plot_contour_var_DJF, 
                              colors=['gray'], levels = pl_DJF_levels, alpha = 0.7, extend='both',
                              transform=ccrs.PlateCarree())

pl_DJF_contours_labels = pl_DJF.axes.clabel(pl_DJF_contours, inline=True, fontsize=16)

# COLOR PLOT
pl_MAM = mslp_plot_color_var_MAM.plot(transform=ccrs.PlateCarree(), 
                                   vmin=mslp_plot_color_min, vmax=mslp_plot_color_max,
                                     add_colorbar=False, ax = axs[1])
pl_MAM.axes.set_extent(mslp_plot_extents, 
              ccrs.PlateCarree())
pl_MAM.set_cmap(mslp_plot_colorscheme)
pl_MAM.axes.coastlines(linewidth = 2.0)

# CONTOUR PLOT
pl_MAM_levels = list(range(mslp_plot_contour_min, mslp_plot_contour_max, mslp_plot_contour_levels))

pl_MAM_contours = pl_MAM.axes.contour(mslp_plot_contour_var_DJF['longitude'], mslp_plot_contour_var_DJF['latitude'], 
                              mslp_plot_contour_var_MAM, 
                              colors=['gray'], levels = pl_MAM_levels, alpha = 0.7, extend='both',
                              transform=ccrs.PlateCarree())

pl_MAM_contours_labels = pl_MAM.axes.clabel(pl_MAM_contours, inline=True, fontsize=16)

# COLOR PLOT
pl_JJA = mslp_plot_color_var_JJA.plot(transform=ccrs.PlateCarree(), 
                                   vmin=mslp_plot_color_min, vmax=mslp_plot_color_max,
                                     add_colorbar=False, ax = axs[2])
pl_JJA.axes.set_extent(mslp_plot_extents, 
              ccrs.PlateCarree())
pl_JJA.set_cmap(mslp_plot_colorscheme)
pl_JJA.axes.coastlines(linewidth = 2.0)

# CONTOUR PLOT
pl_JJA_levels = list(range(mslp_plot_contour_min, mslp_plot_contour_max, mslp_plot_contour_levels))

pl_JJA_contours = pl_JJA.axes.contour(mslp_plot_contour_var_DJF['longitude'], mslp_plot_contour_var_DJF['latitude'], 
                              mslp_plot_contour_var_JJA, 
                              colors=['gray'], levels = pl_JJA_levels, alpha = 0.7, extend='both',
                              transform=ccrs.PlateCarree())

pl_JJA_contours_labels = pl_JJA.axes.clabel(pl_JJA_contours, inline=True, fontsize=16)

# COLOR PLOT
pl_SON = mslp_plot_color_var_SON.plot(transform=ccrs.PlateCarree(), 
                                   vmin=mslp_plot_color_min, vmax=mslp_plot_color_max,
                                     add_colorbar=False, ax = axs[3])
pl_SON.axes.set_extent(mslp_plot_extents, 
              ccrs.PlateCarree())
pl_SON.set_cmap(mslp_plot_colorscheme)
pl_SON.axes.coastlines(linewidth = 2.0)

# CONTOUR PLOT
pl_SON_levels = list(range(mslp_plot_contour_min, mslp_plot_contour_max, mslp_plot_contour_levels))

pl_SON_contours = pl_SON.axes.contour(mslp_plot_contour_var_DJF['longitude'], mslp_plot_contour_var_DJF['latitude'], 
                              mslp_plot_contour_var_SON, 
                              colors=['gray'], levels = pl_SON_levels, alpha = 0.7, extend='both',
                              transform=ccrs.PlateCarree())

pl_SON_contours_labels = pl_SON.axes.clabel(pl_SON_contours, inline=True, fontsize=16)

# general plotting settings + extra lines

axs[0].set_title("(e) DJF", fontsize=24, weight='bold')
axs[1].set_title("(f) MAM", fontsize=24, weight='bold')
axs[2].set_title("(g) JJA", fontsize=24, weight='bold')
axs[3].set_title("(h) SON", fontsize=24, weight='bold')

# plot basin bounds
pl_bound_1 = axs[0].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_2 = axs[0].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

pl_bound_11 = axs[1].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_21 = axs[1].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

pl_bound_12 = axs[2].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_22 = axs[2].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

pl_bound_13 = axs[3].scatter(glacier_bound_1["longitude"], glacier_bound_1["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )
pl_bound_23 = axs[3].scatter(glacier_bound_2["longitude"], glacier_bound_2["latitude"],
     color='black', marker='o', s=0.7,
     transform=ccrs.PlateCarree()
     )

gl1 = axs[0].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl2 = axs[1].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl3 = axs[2].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)
gl4 = axs[3].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)

gl1.xlabel_style = {'size': 20}
gl1.ylabel_style = {'size': 20}
gl1.xlabels_top = False
gl1.ylabels_right = False

gl2.xlabel_style = {'size': 20}
gl2.ylabel_style = {'size': 20}
gl2.xlabels_top = False
gl2.ylabels_left = False
gl2.ylabels_right = False

gl3.xlabel_style = {'size': 20}
gl3.ylabel_style = {'size': 20}
gl3.xlabels_top = False
gl3.ylabels_left = False
gl3.ylabels_right = False

gl4.xlabel_style = {'size': 20}
gl4.ylabel_style = {'size': 20}
gl4.xlabels_top = False
gl4.ylabels_left = False

gl1.ylocator = mticker.FixedLocator([-80, -70, -60])
gl1.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl2.ylocator = mticker.FixedLocator([-80, -70, -60])
gl2.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl3.ylocator = mticker.FixedLocator([-80, -70, -60])
gl3.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl4.ylocator = mticker.FixedLocator([-80, -70, -60])
gl4.xlocator = mticker.FixedLocator([-120, -60, 60, 0, 120, 180])

gl1.rotate_labels = False
gl2.rotate_labels = False
gl3.rotate_labels = False
gl4.rotate_labels = False

#gl_labels = axs[0].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=True)


fig.subplots_adjust(wspace=0.01,hspace=0.01)

cbar_ax = fig.add_axes([0.20, 0.35, 0.60, 0.02])
fig.colorbar(pl_DJF, orientation="horizontal", cax=cbar_ax, extend='both').set_label(label='MSLP anomaly (hPa)',size=24,weight='bold')
cbar_ax.tick_params(labelsize=24)

if basin == "amery":
    plt.savefig(parent_dir+"amery_mslp_event_composites.png")
elif basin == "amundsen":
    plt.savefig(parent_dir+"amundsen_mslp_event_composites.png")

plt.show()
plt.close()


# In[ ]:





# In[ ]:




