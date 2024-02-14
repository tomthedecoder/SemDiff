import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import imageio 
import cartopy 
import torch 
import cartopy.crs as ccrs
import xarray as xr 
from Statistics import Statistics
from torch.nn.functional import mse_loss  

def sl_arr(a, coords, s0, s1):
    if (s0 is not None and s1 is not None):
        a = xr.concat([xr.DataArray(x.squeeze(), coords) for x in a], dim='time').expand_dims(dim='channel', axis=1)
        return a.interp(lat=s0, lon=s1, method='nearest').values
    else: 
        return a

def to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def set_dl_trns(dl, trns):
     dl.dataset.set_trns(trns)
        
def set_lr_trns(dl, trns):
     dl.dataset.set_lr_trns(trns)
        
def change_pri_trainable(model, status=False):
    for param in model.parameters():
        param.requires_grad = status
    model.tau.requires_grad = not status
    
def save_preds(config, eval_dl, model, krepeats, path='/nesi/project/niwa03712/bailiet/preds', flag=''):
    # save a (K, T, 1, N, M) array of predictions to memory
    
    model.eval()
    predictions = [] 
    #model.to(config.device)
    for xlr, _ in eval_dl:
        with torch.no_grad():
            xlr = xlr.to(config.device)
            predictions.extend(to_numpy(model(xlr).float()))
        
    predictions = np.array(predictions)
    np.save(f'{path}_{flag}', predictions)
    
    return predictions 

def train_model(config, model, train_dl, eval_dl, num_epochs=10, criterion=nn.MSELoss(), flag='', 
                val_criterion=mse_loss, optimiser=None, do_save0=True):
    
    mv_set = [] 
    model = model.to(config.device)
    if optimiser is None:
        optimiser = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    train_loss, val_loss = [], []
    work_dir = config.work_dir
    min_val_loss = 0 
    for epoch in range(num_epochs):
        model.train()
        train_loss.append([])
        for batch_idx, batch in enumerate(train_dl):
            xlr, xhr = batch[0].to(config.device), batch[1].to(config.device)
            optimiser.zero_grad()
            pred_xhr = model(xlr).float()
            xhr = xhr.float()
            loss = criterion(pred_xhr, xhr)         
            loss.backward()
            optimiser.step()
            train_loss[-1].append(loss.item())
        train_loss[-1] = np.mean(train_loss[-1])
                        
        with torch.no_grad():
            val_loss.append([]) 
            for xlr, xhr in eval_dl:
                xlr = xlr.to(config.device)
                xhr = xhr.to(config.device)
                pred_xhr = model(xlr)
                loss = val_criterion(pred_xhr, xhr)
                val_loss[-1].append(loss.item())
            val_loss[-1] = np.mean(val_loss[-1])
            
        # save minimum validation loss and use this model in place of subsequent 
        if epoch == 0 or min_val_loss > val_loss[-1]:
            min_val_loss = val_loss[-1]
            print('Epoch, Min loss', epoch+1, min_val_loss)
            torch.save(model, f'{work_dir}/{flag}.pth')
        
        mv_set.append(min_val_loss)
    
    best_model = torch.load(f'{work_dir}/{flag}.pth')
    return best_model, mv_set 

def get_loc_dict():
    def alter_ld(loc_dict, ws=10, ns=10):
        for ln, (lat, lon) in loc_dict.items():
            if lat is None or lon is None:
                continue 
                
            # distance between lat and lon coords on the data grid 
            lat_step = 0.10794 * ws//2
            lon_step = 0.10733 * ws//2
            
            lats  = np.linspace(lat-lat_step, lat+lat_step, ns)
            lons = np.linspace(lon-lon_step, lon+lon_step, ns)
            loc_dict[ln] = [lats, lons]

    # loc dict for variety of locations
    loc_dict = {
                    'New Zealand': [None, None],
                    'Auckland': [-36.8509, 174.7645],
                    "Wellington": [-41.2924, 174.7787],
                    'Tongariro': [-39.1928, 175.5938],
                    'Napier': [-39.4893, 176.9192],
                    'Greymouth': [-42.4614, 171.1985],
                    'Mt. Cook': [-43.5950, 170.1418]
               }
    alter_ld(loc_dict)
    
    return loc_dict 