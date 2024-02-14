import matplotlib.pyplot as plt 
import numpy as np 
import xarray as xr
import cartopy.crs as ccrs 

_dir = '/nesi/project/niwa03712/bailiet'

semdiff = np.load(f'{_dir}/preds__SemSDM2_cc_repeat0.npy')
unet = np.load(f'{_dir}/preds_UNET.npy')[0:len(semdiff)]
sdm = np.load(f'{_dir}/preds_SemSDM2.npy')[0:len(semdiff)]
sdm_cc = np.load(f'{_dir}/preds_SDM_cc_repeat0.npy')
semdiff_ncc = np.load(f'{_dir}/preds_SemDiff.npy')

# get gt 
gt = xr.open_dataset("/nesi/nobackup/niwa03712/shared_paths/high-res-data/ACCESS-CM2_historical_precip.nc").to_array().squeeze()
gt = gt[19419:, 0:-4, 0:-11]


coords = gt.coords
semdiff = xr.DataArray(semdiff.squeeze(), coords)
unet = xr.DataArray(unet.squeeze(), coords)

semdiff = np.mean(semdiff, axis=0)
unet = np.mean(unet, axis=0)
sdm = np.mean(sdm, axis=0)
sdm_cc = np.mean(sdm_cc, axis=0)
semdiff_ncc = np.mean(semdiff_ncc, axis=0)
gt = np.mean(gt, axis=0)
coords = gt.coords

semdiff = xr.DataArray(semdiff.squeeze(), coords)
unet = xr.DataArray(unet.squeeze(), coords)
sdm =  xr.DataArray(sdm.squeeze(), coords)
sdm_cc = xr.DataArray(sdm_cc.squeeze(), coords)
semdiff_ncc = xr.DataArray(semdiff_ncc.squeeze(), coords)


cmap = 'Blues'
vmax = None
titles = ['UNET', 'SemDiff(cc)']#,# 'Ground Truth', 'ScaleSDM', 'ScaleSDM(cc)', 'SemDiff']
to_plots = [unet, semdiff, gt, sdm, sdm_cc, semdiff_ncc]
#figs = [plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=171.77))) for _ in range(3)]

for title, array in zip(titles, to_plots):
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=171.77)))
    array.plot(ax=ax, cmap=cmap, vmax=8, transform=ccrs.PlateCarree())
    
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.coastlines('10m')
    ax.set_title(title)

    plt.savefig(f'/home/bailiet/DDM/mlde/{title}_spt_plot.pdf')
    
    fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree(central_longitude=171.77)))
    array = array.groupby("time.year").max().mean("year") 
    array.plot(ax=ax, cmap='BrBG', vmax=200, transform=ccrs.PlateCarree())
    
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.coastlines('10m')
    ax.set_title(title)

    plt.savefig(f'/home/bailiet/DDM/mlde/{title}_spt_plot.pdf') 