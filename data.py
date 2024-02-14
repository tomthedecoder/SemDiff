from torch.utils.data import Dataset, DataLoader   
import torchvision.datasets as datasets 
from torchvision.transforms import ToTensor 
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda
from functools import partial 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np 
import xarray as xr 
import torch 
from sklearn.mixture import GaussianMixture
from typing import * 
from skimage.feature import peak_local_max 
from plot import show_samples 
from utils import * 
from itertools import product 
from prior_model import CascadedModelWrapper

def load_cascade(config, dls=None, use_chain=True, path='/nesi/project/niwa03712/bailiet/add_work', samplers=None, dataloaders=None):
    save_all = samplers is not None 
    past_models = [] 
    cas_model = None
    for n in range(len(config.prior.taus)-1):
        name = f'sem_{n}_epoch' if use_chain else f'ABsem_{n}_epoch'
        model = torch.load(f'{path}/{name}.pth')
        model.inference_mode = True 
        cas_model = CascadedModelWrapper(config.device, model, past_models, use_chain=config.prior.use_chain)
        if save_all:
            samplers[name] = cas_model 
            dataloaders.append(dls['eval_pri'][n])
        past_models.append(model)
    cas_model = cas_model.eval()
    if save_all:
        return samplers, dataloaders
    else:
        return cas_model

class Transform:
    def __init__(self, trfns, device):
        self.trfns = lambda x: trfns(x.unsqueeze(0)).squeeze().unsqueeze(0)
        self.device = device 
    
    def __call__(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            x = torch.concat((x, self.trfns(x)), dim=0)
        return x 

class PairedDataset(Dataset):
    def __init__(self, cx_data, x_data, label: int = None, coords=None, hr_transform=None, lr_transform=None):
        super().__init__()
        
        self.x_data = x_data
        self.cx_data = cx_data
        self.label = torch.tensor(label) if label is not None else None  
        self.coords = coords 
        self.hr_transform = hr_transform
        self.lr_transform = lr_transform 
    
    def __getitem__(self, idx):
        xhr = self.hr_transform(self.x_data[idx]) if (self.hr_transform is not None) else self.x_data[idx]
        xlr = self.lr_transform(self.cx_data[idx]) if (self.lr_transform is not None) else self.cx_data[idx]
        if self.label is not None: 
            return (self.cx_data[idx], self.x_data[idx]), self.label
        else:
            return xlr, xhr
    
    def __len__(self):
        return len(self.x_data)
    
    def set_trns(self, new_trns):
        self.hr_transform = new_trns
    
    def set_lr_trns(self, new_lr_trns):
        self.lr_transform = new_lr_trns
    
def plot_data_samples(config, cond_train, nplots=4, path='/home/bailiet/DDM/mlde/slurm/data'):
    for n in range(config.boosting.nmodels):
        for k, ((im, hr_im), label) in enumerate(cond_train[n]):
            if k > nplots:
                break
                
            for t, pred_im in enumerate(im.squeeze()):
                plt.figure()
                plt.imshow(pred_im, cmap='Blues')
                plt.savefig(f'{path}/LR label: {n}, number: {k}, pred index: {t}')
                plt.close()
            plt.figure()
            plt.imshow(hr_im.squeeze(), cmap='Blues')
            plt.savefig(f'{path}/HR label: {n}, number: {k}')
            plt.close()
 
# from QB section copy + paste 

def data_between(y, q_low, X=None, q_high=None):
    # get the datapoints within a quantile range, measured from cumculative sum on y
    # if q_high is None, then select only the y closest to q_low
    sum_array, _ = torch.sort(torch.sum(y, axis=(-1, -2, -3)))
    N = len(sum_array)
    # collect indexes for low and high sum boundaries 
    # round of i_low, i_high is the 'nearest' method, effectively qval=0.7666 is 0.76 for example
    # interpolation is not feasiable in this setting
    i_low = min(int(q_low * N), N - 1)
    if q_high is not None:
        i_high = min(int(q_high * N), N - 1)
        
    y_low_sum = sum_array[i_low]
    if q_high is None:
        y_high_sum = y_low_sum
    else:
        y_high_sum = sum_array[i_high] 
    
    y_low_sum = torch.round(y_low_sum)
    y_high_sum = torch.round(y_high_sum)
    mask = [True if y_low_sum <= torch.squeeze(torch.round(torch.sum(yi, axis=(-1, -2, -3)))) <= y_high_sum else False for yi in y]

    if X is not None:
        return X[mask], y[mask]   
    else:
        return y[mask]

# discretise the dataset, taking y sample to bin index  
def bin_y_var(y, Q_ranges):
    bins = []
    
    # get bin limits 
    for ql, qh in Q_ranges:
        yq = data_between(y, qh) 
        bins.append(torch.sum(yq))

    # get quantile ranges 
    quantiles = []
    for y_sample in y:
        ys_sum = torch.sum(y_sample)
        bin_index = len(bins) - 1
        for n, bin_thrs in enumerate(bins):
            if ys_sum <= bin_thrs:
                bin_index = n
                break

        quantiles.append(bin_index)        
    
    return quantiles

# get quantile intervals 
# perform gmm clustering over sum of each sample 
def get_Q_ranges(y, nmodels):
    N = len(y)
    y = np.array(y)
    sum_y = np.sum(y, axis=(-1, -2))
    gmm = GaussianMixture(n_components=nmodels, n_init=100).fit(sum_y)
    counts = np.sort([np.sum(np.where(sum_y <= y_upper, 1.0, 0.0)) for y_upper in gmm.means_])
    
    Q_ranges = [c/N for c in counts]
    # create quantile intervals 
    Q_ranges.insert(0, 0)
    Q_ranges[-1] = 1.0   # expand largest quantile to end point 
    Q_ranges = [(Q_ranges[n], Q_ranges[n+1]) for n in range(len(Q_ranges) - 1)] 
    
    return Q_ranges

def intensity_mask(x, eps=0.01):
    # fits guassian to x, if distribution normal, then mask out these values 
    mu = np.mean(x, axis=(-1, -2))
    std = np.std(x, axis(-1, -2))
    
    is_normal = (np.abs(np.mean(mu)) < eps and np.abs(1 - std) < eps) 
    is_normal = is_normal.squeeze()
    x[is_normal] = 0.0
    
    return x 

def find_focal_point(config, xlr, xhr):            
    # get segments of the prior image  
    
    def get_bounds(coords, ws, mx_bd, i):
        # returns bounds (x or y) of given coords array 
        bound = lambda x, b, s: np.where(x > b if s == '>' else x < b, b, x)
        bds = np.stack((bound(coords[:, i], 0, '<'), bound(coords[:, i]+ws, mx_bd, '>'))).transpose().squeeze()
        # correct bound if sample too small 
        for n, (b0, b1) in enumerate(bds):
            diffw = config.fusion.ws - (b1 - b0)
            if diffw > 0:  # correct bound? 
                if b0 == 0:
                    b1 += diffw
                if b1 == mx_bd:
                    b0 -= diffw
            bds[n] = (b0, b1) 
            
        return bds 
    
    def extract(im, coords, ws):
        # use bounds to get image segment 
        seg_tensor = None 
        for n, (x0, y0) in enumerate(coords):
            seg_im = im[n,:,x0:x0+ws,y0:y0+ws]
            _max = torch.max(seg_im)
            seg_im = seg_im / _max 
            seg_tensor = seg_im if seg_tensor is None else torch.concat((seg_tensor, seg_im), dim=1) # this line 
        seg_tensor = torch.unsqueeze(seg_tensor, 1)
        
        return seg_tensor

    # if optimisation method cannot find maxima (extreme) then p.l.m. returns []  
    # will only work for 1 peak (adjacent pixels) 
    n, m = config.data.shape 
    ws = config.fusion.ws
    xhr = xhr.to(config.device)
    nkerx, nkery = n-ws+1,m-ws+1 
    coords = [] 
    k = 0
    for xi in xhr:
        xi = torch.unsqueeze(xi, 0)
        seg = F.unfold(xi, kernel_size=ws)
        _sum = torch.sum(seg, axis=-2)
        values, ker_ind = torch.topk(_sum, 1, axis=-1, sorted=True) # top k for multiple
        coords.extend([(ki//nkerx,ki%nkery) for val, ki in zip(values, ker_ind)]) #if val > 0.5])
        k += 1

    #coords = filter_points(coords, config.fusion.min_dist)
    
    mask = []
    boxed_xhr = extract(xhr, coords, ws)
    boxed_xlr = extract(xlr, coords, ws)

    return boxed_xlr, boxed_xhr, coords

# end copy and paste 

def open_data(config):
    # opens the dataset 
    # converts the loaded data into custom dataset, can have low res, high res and label or just the last two 
    # (lat=config.data.coarse_factor, lon=config.data.coarse_factor, boundary='pad').mean().values
    
    def coarsen(xhr):
        # coarsen by LR and normalise this by maximise by LR image
        
        xlr = F.avg_pool2d(xhr, kernel_size=config.data.coarse_factor)
        xlr = F.interpolate(xlr, xhr.shape[2:], mode='nearest')
        m = torch.amax(xlr, dim=(-1, -2), keepdim=True)
        xlr = xlr / m
        
        return xlr, m 
    
    def cond_ds(ylr, yhr, Q_splits, bs):
        # add to conditional classes dicts, class k -> dataset of (xlr, xhr, k) given kth quantile  
        labels = bin_y_var(yhr, Q_splits)
        
        # need way to take ylr, yhr, l and only select 
        cond = {l: [[], []] for l in range(config.boosting.nmodels)}
        for yl, yh, l in zip(ylr, yhr, labels):
            cond[l][0].append(yl)
            cond[l][1].append(yh)
        
        for l, (yl, yh) in cond.items():
            cond[l] = DataLoader(PairedDataset(yl, yh, l), batch_size=bs, shuffle=False)
        
        return cond 
    
    def make_all_ds(yhr, bs, trfns=lambda x:x, Q_splits=None, eval_set=False, multi_class=False):
        # get splits for extrema boundaries  
        ret_Q = Q_splits is None 
        Q_splits = get_Q_ranges(yhr, config.boosting.nmodels) if ret_Q else Q_splits 
       
        
        # semantic features of yhr 
        taus = config.prior.taus
        norm_yhr = yhr / np.max(yhr, axis=(-1, -2), keepdims=True)
        previous_out = torch.zeros(yhr.shape)
        cas_data = [] 
        for tau in zip(taus[:-1]):
            sem_yhr = np.piecewise(norm_yhr, [norm_yhr < tau, (norm_yhr >= tau) & (norm_yhr <= taus[-1])], [lambda y: 0, lambda y: 1])
            sem_yhr = torch.from_numpy(sem_yhr)
            
            if eval_set or multi_class:
                previous_out = previous_out + sem_yhr
                cas_data.append(previous_out)
            else:
                cas_data.append(sem_yhr)
                        
        # low resolution features by avg pooling 
        yhr = torch.from_numpy(yhr)
        ylr, m = coarsen(yhr)
        
        ylr_90, yhr_90 = data_between(yhr, 0.9, X=ylr, q_high=1.0)

        full_ds = DataLoader(PairedDataset(ylr, yhr), batch_size=bs, shuffle=False)
        dsem_ds = DataLoader(PairedDataset(ylr, yhr, lr_transform=trfns), batch_size=bs, shuffle=False)
        y90_ds = DataLoader(PairedDataset(ylr_90, yhr_90), batch_size=bs, shuffle=False)
        
        for n, sem_yhr in enumerate(cas_data):
             cas_data[n] = DataLoader(PairedDataset(ylr, sem_yhr), batch_size=bs, shuffle=False) 
                
        if ret_Q:
            return full_ds, cas_data, dsem_ds, y90_ds, Q_splits 
        else:
            return full_ds, cas_data, dsem_ds, y90_ds
            
    # get coords for xarray reconstruction later on 
    full_ds = xr.open_dataset(config.data.file_path).to_array()
    coords = full_ds.coords
    del coords['time']
    del coords['variable']
    coords = {'lat': coords['lat'].values[0:-4], 'lon': coords['lon'].values[0:-11]}

    # convert xr.ds to torch.data loaders 
    full_ds = np.expand_dims(full_ds.values.squeeze(), axis=1)
    full_ds = full_ds[:config.data.nsamples,:,0:-4,0:-11]

    # prior_model, if not yet trained needs work 
    #pri_model = load_cascade(config, use_chain=config.prior.use_chain)
    pri_model = lambda x: x
    #trfns = Transform(pri_model.to(config.device), config.device)
    trfns = Transform(pri_model, config.device)
    
    # dataloaders for train and test #19417
    
    # split points 
    mc = config.prior.multi_class
    tsp = int(config.data.ptrain*len(full_ds))
    vsp = int((config.data.peval+config.data.ptrain)*len(full_ds))
    trn_full, trn_psem, trn_dsem, trn90, Q_splits = make_all_ds(full_ds[:tsp], config.training.batch_size, trfns, eval_set=False, multi_class=mc)
    val_full, val_psem, val_dsem, val90 = make_all_ds(full_ds[tsp:vsp], 1, trfns, Q_splits, eval_set=True, multi_class=mc) 
    evl_full, evl_psem, evl_dsem, evl90 = make_all_ds(full_ds[vsp:], 1, trfns, Q_splits, eval_set=True, multi_class=mc) 

    return {'train': trn_full, 'eval': evl_full, 
            'train_pri': trn_psem, 'eval_pri': evl_psem,
            'train_sdiff': trn_dsem, 'eval_sdiff': evl_dsem,
            'val': val_full, 'val_pri': val_psem, 'val_sdiff': val_dsem,
            'val90': val90, 'eval90': evl90, 'train90': trn90,
            'coords': coords}