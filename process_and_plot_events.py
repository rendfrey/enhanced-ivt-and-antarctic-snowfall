#!/usr/bin/env python
# coding: utf-8

# In[31]:


import xarray as xr
import cartopy.crs as ccrs
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


# In[32]:


#parent_dir = "/nfs/turbo/clasp-aepayne/ERA5/rendfrey/southern_hemisphere/"
parent_dir = "/raid01/rendfrey/IVT/event_only/"

ivt_east_dir = parent_dir+"vertical_integral_of_eastward_water_vapour_flux/"
ivt_north_dir = parent_dir+"vertical_integral_of_northward_water_vapour_flux/"
mslp_dir = parent_dir+"mean_sea_level_pressure/"

monthly_mean_dir = "/nfs/turbo/clasp-aepayne/ERA5/rendfrey/monthly_means/"


# In[33]:


basin_set = [21, 22]
#basin_set = [9, 11]
#basin_set = [13]

seasons = ["DJF", "MAM", "JJA", "SON"]
season = seasons[3]

percentile_cutoff = 95
threshold = percentile_cutoff / 100.


# In[34]:


#ds = xr.open_dataset(parent_dir+"basin_specific/high_IVT_events_"+season+"_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[35]:


#ds


# In[36]:


#ds["Duration"][97:110].mean()/3600000000000


# In[37]:


# open IVT files
ivt_ds = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+season+".nc") # total IVT calculated by a previous script
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


# In[38]:


def get_season(ds):
    ds = ds.sel(time=(ds['time.season']==season))
    return ds


# In[39]:


# total IVT and its percentiles are already calculated by season, but the rest of the data files are not
# therefore, cut the relevant season out from the files

ivt_east_ds = get_season(ivt_east_ds)
ivt_north_ds = get_season(ivt_north_ds)
ivt_monthly_means = get_season(ivt_monthly_means)

mslp_ds = get_season(mslp_ds)
mslp_monthly_means = get_season(mslp_monthly_means)


# In[40]:


# take the time average of the monthly means

ivt_monthly_means = ivt_monthly_means.mean(dim="time")

mslp_monthly_means = mslp_monthly_means.mean(dim="time")


# In[41]:


# combine datasets of the eastward and northward WV transport components
ivt_combined_ds = xr.merge([ivt_east_ds, ivt_north_ds])


# In[42]:


# compute data arrays of IVT anomaly components
da_ivt_eastward_anomaly = ivt_combined_ds["p71.162"] - ivt_monthly_means["p71.162"]
da_ivt_northward_anomaly = ivt_combined_ds["p72.162"] - ivt_monthly_means["p72.162"]

# merge ivt anomaly components into one dataset 
ivt_combined_anomaly_ds = xr.merge([da_ivt_eastward_anomaly, da_ivt_northward_anomaly])

# calculate ivt anomaly magnitude
da_ivt_anomaly = np.sqrt(da_ivt_northward_anomaly**2 + da_ivt_eastward_anomaly**2)
da_ivt_anomaly.attrs["units"] = "kg m^-1 s^-1"

# compute data array of the mslp anomaly
da_mslp_anomaly = mslp_ds["msl"] - mslp_monthly_means["msl"]

# set mslp units to hPa
da_mslp = mslp_ds["msl"] / 100.
da_mslp_anomaly = da_mslp_anomaly / 100.


# In[43]:


# get data set filtered down to only the time steps with interesting IVT over the chosen glacier basin
if len(basin_set) == 1:

    filtered_ds = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+season+"_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist = filtered_ds.time.values


# In[44]:


# the following combines two event lists
if len(basin_set) == 2:

    # get all "A" basin times
    filtered_ds_A = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+season+"_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_A = filtered_ds_A.time.values

    # get all "B" basin times
    filtered_ds_B = xr.open_dataset(parent_dir+"total_water_vapour_flux/vertical_integral_of_total_wv_flux_"+season+"_basin_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.nc")
    timelist_B = filtered_ds_B.time.values

    # following line says "what does the left set have that the right does not?"
    times_B_excl_A = set(timelist_B) - set(timelist_A) # times in B that are not in A

    times_B_excl_A_array = np.array(list(times_B_excl_A)) # change set into array
    times_A_and_B = np.concatenate((times_B_excl_A_array, timelist_A), axis=0) # combine B and A time coordinates

    big_list = times_A_and_B[:(len(times_B_excl_A_array)+len(timelist_A)-1)]
    all_times = np.sort(big_list)
    timelist = all_times


# In[45]:


# save time coordinates as timearray
if len(basin_set) == 1:
    fname = "event_time_coordinates_basins_"+str(season)+"_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile.txt"
if len(basin_set) == 2:
    fname = "event_time_coordinates_basins_"+str(season)+"_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile.txt"
for time in timelist:
    with open(fname, 'a') as the_file:
        the_file.write(str(time)+'\n')


# In[46]:


def filter_by_time_list(data, time_coords):
    data = data.sel(time=time_coords)
    return data


# In[47]:


# open AR tag files
ds_ar = xr.open_mfdataset(parent_dir+"/AR_tags/*.nc")
ds_ar = get_season(ds_ar)
# slice it into a box
ds_ar_box = ds_ar.sel(lat=slice(-80, -72)).sel(lon=slice(-113, -88)) 
# stack the coords
ds_ar_box_stacked = ds_ar_box.stack(z=["time", "lat", "lon"])
# filter by AR tags
ds_ar_box_filtered = ds_ar_box_stacked.where(ds_ar_box_stacked["ar_binary_tag"]>0).dropna(dim="z")
# unstack the filtered-by-AR-tag data
ds_ar_box_filtered = ds_ar_box_filtered.unstack()
# slice down to just cloudsat events
ds_ar_box_cloudsat_events = ds_ar_box_filtered.sel(time=slice("2007-01-01", "2011-01-01"))


# In[48]:


AR_timelist = ds_ar_box_cloudsat_events["time"].values


# In[49]:


# calculate start, end times and duration for each event.

AR_start_times = []
AR_end_times = []
AR_durations = []

ticks = 0
for i in range(len(ds_ar_box_cloudsat_events.time.values)-1):
    
    x = ds_ar_box_cloudsat_events.isel(time=i).time.values
    y = ds_ar_box_cloudsat_events.isel(time=i+1).time.values
    
    # if an event-flagged timestamp (x) is more than 24 hours away from the next one (y),
    # count the timestamp x as the end of that event. 3600000000000 = nanoseconds per hour
    
    if (y - x) > (3600000000000 * 24):
        event_end_time = ds_ar_box_cloudsat_events.isel(time=i).time.values
        event_start_time = ds_ar_box_cloudsat_events.isel( time=(i-ticks) ).time.values
        
        AR_end_times.append(event_end_time)
        AR_start_times.append(event_start_time)
        
        duration = int( (event_end_time - event_start_time)/3600000000000 ) + 3
        
        AR_durations.append(duration)
        
        ticks = 0
        
    else:
        ticks += 1


# In[50]:


# filter all of the data by the time list

ivt_combined_ds_filtered_all = filter_by_time_list(ivt_combined_ds, timelist)

ivt_ds_filtered_all = filter_by_time_list(ivt_ds, timelist)
da_ivt_filtered_all = ivt_ds_filtered_all["Vertical integral of total water vapor flux"]

da_ivt_percentile_filtered_all = ivt_ds_filtered_all["Vertical integral of total water vapor flux percentile rank"]
#da_ivt_percentile_filtered_all *= 100 

ivt_combined_anomaly_filtered_all = filter_by_time_list(ivt_combined_anomaly_ds, timelist)
da_ivt_anomaly_filtered_all = filter_by_time_list(da_ivt_anomaly, timelist)

da_mslp_filtered_all = filter_by_time_list(da_mslp, timelist)

da_mslp_anomaly_filtered_all = filter_by_time_list(da_mslp_anomaly, timelist)


# In[51]:


# calculate start, end times and duration for each event.

event_start_times = []
event_end_times = []
durations = []

ticks = 0
for i in range(len(ivt_ds_filtered_all.time.values)-1):
    
    x = ivt_ds_filtered_all.isel(time=i).time.values
    y = ivt_ds_filtered_all.isel(time=i+1).time.values
    
    # if an event-flagged timestamp (x) is more than 24 hours away from the next one (y),
    # count the timestamp x as the end of that event. 3600000000000 = nanoseconds per hour
    
    if (y - x) > (3600000000000 * 24):
        event_end_time = ivt_ds_filtered_all.isel(time=i).time.values
        event_start_time = ivt_ds_filtered_all.isel( time=(i-ticks) ).time.values
        
        event_end_times.append(event_end_time)
        event_start_times.append(event_start_time)
        
        duration = int( (event_end_time - event_start_time)/3600000000000 ) + 3
        
        durations.append(duration)
        
        ticks = 0
        
    else:
        ticks += 1


# In[52]:


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


# In[53]:


ivt_combined_ds.sel(time=timelist)


# In[54]:


# filter all of the data by the AR list
"""
AR_ivt_combined_ds_filtered_all = filter_by_time_list(ivt_combined_ds, AR_timelist)

AR_ivt_ds_filtered_all = filter_by_time_list(ivt_ds, AR_timelist)
AR_da_ivt_filtered_all = AR_ivt_ds_filtered_all["Vertical integral of total water vapor flux"]

AR_da_ivt_percentile_filtered_all = AR_ivt_ds_filtered_all["Vertical integral of total water vapor flux percentile rank"]
AR_da_ivt_percentile_filtered_all *= 100 

AR_ivt_combined_anomaly_filtered_all = filter_by_time_list(ivt_combined_anomaly_ds, AR_timelist)
AR_da_ivt_anomaly_filtered_all = filter_by_time_list(da_ivt_anomaly, AR_timelist)

AR_da_mslp_filtered_all = filter_by_time_list(da_mslp, AR_timelist)

AR_da_mslp_anomaly_filtered_all = filter_by_time_list(da_mslp_anomaly, AR_timelist)
"""


# In[55]:


if len(basin_set) == 1:
    event_times.to_netcdf(parent_dir+"high_IVT_events_"+season+"_basin_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")

if len(basin_set) == 2:
    event_times.to_netcdf(parent_dir+"high_IVT_events_"+season+"_basins_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.nc")


# In[56]:


da_ivt_percentile_filtered_all = da_ivt_percentile_filtered_all * 100


# In[57]:


### IVT plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events

ivt_plot_color_var = da_ivt_percentile_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_color_var.attrs["units"] = "Percentile rank of IVT"
ivt_plot_color_min = 0
ivt_plot_color_max = 100
ivt_plot_extents = [-180, 180, -90, -55]
ivt_plot_colorscheme = 'GnBu'

ivt_plot_quiver_var = ivt_combined_ds_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_quiver_scale = 1700

ivt_plot_contour_var = da_ivt_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
ivt_plot_contour_var.attrs["units"] = "kg m^-1 s^-1"
ivt_plot_contour_min = 0
ivt_plot_contour_max = 200
ivt_plot_contour_levels = 50


# In[58]:


# IVT plot
figure_size = (20, 25)
proj_type = ccrs.SouthPolarStereo()

# defining the figure
fig = plt.figure(figsize=figure_size)
axs = plt.axes(projection=proj_type, facecolor='w')
#######################################################################

# COLOR PLOT
pl = ivt_plot_color_var.plot(transform=ccrs.PlateCarree(), 
                                   vmin=ivt_plot_color_min, vmax=ivt_plot_color_max,
                                   cbar_kwargs={'label': ivt_plot_color_var.attrs["units"]},)
pl.axes.set_extent(ivt_plot_extents, 
              ccrs.PlateCarree())
pl.set_cmap(ivt_plot_colorscheme)
pl.axes.coastlines(linewidth = 2.0)

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample = ivt_plot_quiver_var.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv = resample.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="black",
                            transform=ccrs.PlateCarree(),
                            scale = ivt_plot_quiver_scale, 
                            )
plt.quiverkey(quiv, 1.00, 1.10, 
              ivt_plot_contour_max, r' '+str(ivt_plot_contour_max,)+' '+ivt_plot_contour_var.attrs["units"], 
              labelpos='N')

# CONTOUR PLOT
pl_levels = list(range(ivt_plot_contour_min, ivt_plot_contour_max, ivt_plot_contour_levels))

pl_contours = pl.axes.contour(ivt_plot_contour_var['longitude'], ivt_plot_contour_var['latitude'], 
                              ivt_plot_contour_var, 
                              colors=['gray'], levels = pl_levels,
                              transform=ccrs.PlateCarree())

pl_contours_labels = pl.axes.clabel(pl_contours, inline=True, fontsize=16)

# general plotting settings + extra lines

#gl60 = pl4.axes.gridlines(draw_labels=False, xlocs=[], ylocs=[-60], 
#                         linewidth=2, color='gray', alpha=0.5, linestyle='-')

#title = str(da_total_ivt.isel(time=siglist).coords['time'].values)
#print(title[:10])
#plt.suptitle(title[:10])
#plt.suptitle("Basin index "+str(basin_indices[0])+"+"+str(basin_indices[1])+" "+str(percentile_cutoff)+"th threshold")

gl = axs.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

axs.set_title('Water vapor transport percentiles')

if len(basin_set) == 1:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basin_"+season+"_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")
if len(basin_set) == 2:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basins_"+season+"_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")

plt.show()
#plt.close()


# In[59]:


# MSLP plot parameters

date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events
mslp_plot_color_var = da_mslp_anomaly_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
mslp_plot_color_var.attrs["units"] = "hPa"
mslp_plot_color_min = -10
mslp_plot_color_max = 10
mslp_plot_extents = [-180, 180, -90, -55]
mslp_plot_colorscheme = 'bwr'

mslp_plot_contour_var = da_mslp_anomaly_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
mslp_plot_contour_min = -20
mslp_plot_contour_max = 20
mslp_plot_contour_levels = 10
###


# In[60]:


# MSLP plot
figure_size = (20, 25)
proj_type = ccrs.SouthPolarStereo()

# defining the figure
fig = plt.figure(figsize=figure_size)
axs = plt.axes(projection=proj_type, facecolor='w')
# COLOR PLOT
pl4 = mslp_plot_color_var.plot(transform=ccrs.PlateCarree(), 
                                   vmin=mslp_plot_color_min, vmax=mslp_plot_color_max,
                                     cbar_kwargs={'label': mslp_plot_color_var.attrs["units"]},)
pl4.axes.set_extent(mslp_plot_extents, 
              ccrs.PlateCarree())
pl4.set_cmap(mslp_plot_colorscheme)
pl4.axes.coastlines(linewidth = 2.0)

# CONTOUR PLOT
pl4_levels = list(range(mslp_plot_contour_min, mslp_plot_contour_max, mslp_plot_contour_levels))

pl4_contours = pl4.axes.contour(mslp_plot_contour_var['longitude'], mslp_plot_contour_var['latitude'], 
                              mslp_plot_contour_var, 
                              colors=['gray'], levels = pl4_levels, alpha = 0.7, extend='both',
                              transform=ccrs.PlateCarree())

pl4_contours_labels = pl4.axes.clabel(pl4_contours, inline=True, fontsize=16)

# general plotting settings + extra lines

#gl60 = pl4.axes.gridlines(draw_labels=False, xlocs=[], ylocs=[-60], 
#                         linewidth=2, color='gray', alpha=0.5, linestyle='-')

#title = str(da_total_ivt.isel(time=siglist).coords['time'].values)
#print(title[:10])
#plt.suptitle(title[:10])
#plt.suptitle("Basin index "+str(basin_indices[0])+"+"+str(basin_indices[1])+" "+str(percentile_cutoff)+"th threshold")

gl = axs.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

axs.set_title('Mean sea level pressure '+season+' anomaly')

if len(basin_set) == 1:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basin_"+season+"_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")
if len(basin_set) == 2:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basins_"+season+"_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")

plt.show()
plt.close()


# In[61]:


quit()


# In[ ]:


date1 = "2007-01-01" # date from which to count events
date2 = "2010-12-31" # date up to which to count events

#date1 = "2008-09-01T18:00:00.000000000"
#date2 = "2008-09-04T18:00:00.000000000"

upper_left_color_var = da_ivt_percentile_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
upper_left_color_var.attrs["units"] = "Percentile rank of IVT"
upper_left_color_min = 0
upper_left_color_max = 100
upper_left_extents = [-180, 180, -90, -55]
upper_left_colorscheme = 'GnBu'

upper_left_quiver_var = ivt_combined_ds_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
upper_left_quiver_scale = 1700

upper_left_contour_var = da_ivt_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
upper_left_contour_var.attrs["units"] = "kg m^-1 s^-1"
upper_left_contour_min = 0
upper_left_contour_max = 200
upper_left_contour_levels = 50
###
upper_right_color_var = da_mslp_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
upper_right_color_var.attrs["units"] = "hPa"
upper_right_color_min = 970
upper_right_color_max = 1030
upper_right_extents = [-180, 180, -90, -55]
upper_right_colorscheme = 'RdYlBu_r'

upper_right_contour_var = da_mslp_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
upper_right_contour_min = 970
upper_right_contour_max = 1030
upper_right_contour_levels = 10
###
#lower_left_color_var = da_ivt_anomaly_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_left_color_var = da_ivt_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_left_color_min = 0
lower_left_color_max = 200
lower_left_extents = [-180, 180, -90, -55]
lower_left_colorscheme = 'GnBu'

lower_left_quiver_var = ivt_combined_ds_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_left_quiver_scale = 1700

#lower_left_contour_var = da_ivt_anomaly_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_left_contour_var = da_ivt_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_left_contour_var.attrs["units"] = "kg m^-1 s^-1"
lower_left_contour_min = 0
lower_left_contour_max = 200
lower_left_contour_levels = 50
###
lower_right_color_var = da_mslp_anomaly_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_right_color_var.attrs["units"] = "hPa"
lower_right_color_min = -10
lower_right_color_max = 10
lower_right_extents = [-180, 180, -90, -55]
lower_right_colorscheme = 'bwr'

lower_right_contour_var = da_mslp_anomaly_filtered_all.sel(time=slice(date1, date2)).mean(dim="time")
lower_right_contour_min = -20
lower_right_contour_max = 20
lower_right_contour_levels = 10
###



# In[ ]:


num_cols = 2
num_rows = 2
figure_size = (20, 25)
proj_type = ccrs.SouthPolarStereo()

# defining the figure
fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols, figsize=figure_size, 
                        subplot_kw=dict(projection=proj_type, facecolor='w') )
#######################################################################

### upper left panel

# COLOR PLOT
pl = upper_left_color_var.plot(transform=ccrs.PlateCarree(), 
                                   vmin=upper_left_color_min, vmax=upper_left_color_max,
                                   cbar_kwargs={'label': upper_left_color_var.attrs["units"]},
                                   ax=axs[0, 0])
pl.axes.set_extent(upper_left_extents, 
              ccrs.PlateCarree())
pl.set_cmap(upper_left_colorscheme)
pl.axes.coastlines()

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample = upper_left_quiver_var.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv = resample.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="gray",
                            transform=ccrs.PlateCarree(),
                            scale = upper_left_quiver_scale, 
                            ax=axs[0, 0])
plt.quiverkey(quiv, 1.00, 1.10, 
              upper_left_contour_max, r' '+str(upper_left_contour_max,)+' '+upper_left_contour_var.attrs["units"], 
              labelpos='N')

# CONTOUR PLOT
pl_levels = list(range(upper_left_contour_min, upper_left_contour_max, upper_left_contour_levels))

pl_contours = pl.axes.contour(upper_left_contour_var['longitude'], upper_left_contour_var['latitude'], 
                              upper_left_contour_var, 
                              colors=['black'], levels = pl_levels,
                              transform=ccrs.PlateCarree())

pl_contours_labels = pl.axes.clabel(pl_contours, inline=True, fontsize=10)

#######################################################################

### upper right panel

# COLOR PLOT
pl2 = upper_right_color_var.plot(transform=ccrs.PlateCarree(), 
                                   vmin=upper_right_color_min, vmax=upper_right_color_max,
                                    cbar_kwargs={'label': upper_right_color_var.attrs["units"]},
                                   ax=axs[0, 1])
pl2.axes.set_extent(upper_right_extents, 
              ccrs.PlateCarree())
pl2.set_cmap(upper_right_colorscheme)
pl2.axes.coastlines()

# CONTOUR PLOT
pl2_levels = list(range(upper_right_contour_min, upper_right_contour_max, upper_right_contour_levels))

pl2_contours = pl2.axes.contour(upper_right_contour_var['longitude'], upper_right_contour_var['latitude'], 
                              upper_right_contour_var, 
                              colors=['black'], levels = pl2_levels,
                              transform=ccrs.PlateCarree())

pl2_contours_labels = pl2.axes.clabel(pl2_contours, inline=True, fontsize=10)
#######################################################################

### lower left panel

# COLOR PLOT
pl3 = lower_left_color_var.plot(transform=ccrs.PlateCarree(), 
                                   vmin=lower_left_color_min, vmax=lower_left_color_max,
                                    cbar_kwargs={'label': lower_left_contour_var.attrs["units"]},
                                   ax=axs[1, 0])
pl3.axes.set_extent(lower_left_extents, 
              ccrs.PlateCarree())
pl3.set_cmap(lower_left_colorscheme)
pl3.axes.coastlines()

# QUIVER PLOT
# downsample lat/lons for quiver plot (don't want to plot too many arrows)
resample2 = lower_left_quiver_var.isel(longitude=slice(None, None, 20),
                              latitude=slice(None, None, 20))
quiv2 = resample2.plot.quiver(x='longitude', y='latitude', u='p71.162', v='p72.162', color="gray",
                            transform=ccrs.PlateCarree(),
                            scale = lower_left_quiver_scale, 
                            ax=axs[1, 0])
plt.quiverkey(quiv2, 1.00, 1.10, 
              lower_left_contour_max, r' '+str(lower_left_contour_max,)+' '+lower_left_contour_var.attrs["units"], 
              labelpos='N')

# CONTOUR PLOT
pl3_levels = list(range(lower_left_contour_min, lower_left_contour_max, lower_left_contour_levels))

pl3_contours = pl3.axes.contour(lower_left_contour_var['longitude'], lower_left_contour_var['latitude'], 
                              lower_left_contour_var, 
                              colors=['black'], levels = pl3_levels,
                              transform=ccrs.PlateCarree())

pl3_contours_labels = pl3.axes.clabel(pl3_contours, inline=True, fontsize=10)
#######################################################################

### lower right panel

# COLOR PLOT
pl4 = lower_right_color_var.plot(transform=ccrs.PlateCarree(), 
                                   vmin=lower_right_color_min, vmax=lower_right_color_max,
                                     cbar_kwargs={'label': lower_right_color_var.attrs["units"]},
                                   ax=axs[1, 1])
pl4.axes.set_extent(lower_right_extents, 
              ccrs.PlateCarree())
pl4.set_cmap(lower_right_colorscheme)
pl4.axes.coastlines()

# CONTOUR PLOT
pl4_levels = list(range(lower_right_contour_min, lower_right_contour_max, lower_right_contour_levels))

pl4_contours = pl4.axes.contour(lower_right_contour_var['longitude'], lower_right_contour_var['latitude'], 
                              lower_right_contour_var, 
                              colors=['black'], levels = pl4_levels, alpha = 0.7,
                              transform=ccrs.PlateCarree())

pl4_contours_labels = pl4.axes.clabel(pl4_contours, inline=True, fontsize=10)

#######################################################################

# general plotting settings + extra lines

#gl60 = pl4.axes.gridlines(draw_labels=False, xlocs=[], ylocs=[-60], 
#                         linewidth=2, color='gray', alpha=0.5, linestyle='-')

#title = str(da_total_ivt.isel(time=siglist).coords['time'].values)
#print(title[:10])
#plt.suptitle(title[:10])
#plt.suptitle("Basin index "+str(basin_indices[0])+"+"+str(basin_indices[1])+" "+str(percentile_cutoff)+"th threshold")

axs[0,0].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)
axs[0,1].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)
axs[1,0].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)
axs[1,1].gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.5, linestyle='--', draw_labels=False)

axs[0,0].set_title('Water vapor transport percentiles')
axs[0,1].set_title('Mean sea level pressure')
axs[1,0].set_title('Water vapor transport '+season+'')
axs[1,1].set_title('Mean sea level pressure '+season+' anomaly')

if len(basin_set) == 1:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basin_"+season+"_"+str(basin_set[0])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")
if len(basin_set) == 2:
    plt.savefig(parent_dir+
                f"diff_event_composite-mean_basins_"+season+"_"+str(basin_set[0])+"_"+str(basin_set[1])+"_"+str(percentile_cutoff)+"th_percentile_IVT.png")

plt.show()
#plt.close()


# In[ ]:




