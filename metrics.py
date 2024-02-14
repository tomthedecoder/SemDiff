import torch 
import torch.nn.functional as F
import csv 
from scipy.stats import binned_statistic, binned_statistic_dd
from plot import *
from torch.nn.functional import mse_loss 
from utils import *  
from functools import partial 
from sklearn.metrics import jaccard_score
from torchmetrics.functional import dice
import numpy as np 
import segmentation_models_pytorch as smp 
from piq import ssim 


def my_ssim(x, y):
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = torch.unsqueeze(torch.unsqueeze(y, dim=0), dim=0)
    x = x / (torch.amax(x, dim=(-1, -2), keepdims=True) + 0.001)
    y = y / (torch.amax(y, dim=(-1, -2), keepdims=True) + 0.001)    
    x = torch.where(torch.isnan(x), 0.0, x)
    y = torch.where(torch.isnan(y), 0.0, y)
             
    return ssim(x, y, kernel_size=3 if x.shape[-1] <= 11 else 11)

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def max_abs(x_true, x_pred):
    res = torch.abs(x_true - x_pred)
    topk_res = torch.topk(res, 10, dim=(-1, -2), keepdim=True)
    
    abs_res = torch.mean(topk_res)
    return abs_res 

def one_hot_encode(x_true, x_pred, nclasses=-1):
    if nclasses == -1:
        nclasses = torch.max(x_true).int().item()+1
        
    x_true = F.one_hot(x_true.long(), nclasses)
    x_pred = F.one_hot(x_pred.long(), nclasses)
    
    return x_true, x_pred, nclasses 


def dice_score(x_true, x_pred, weight=False, eps=0.0001):
    return dice(x_true.long(), x_pred.long())

    
def iou(x_true, x_pred, eps=0.01):
    x_true, x_pred, nclasses = one_hot_encode(x_true, x_pred)
    
    intersection = torch.logical_and(x_true, x_pred)
    union = torch.logical_or(x_true, x_pred)
    
    iou_score = (intersection.sum() + eps) / (union.sum() + eps)
    
    return iou_score

def pixel_accuracy(x_true, x_pred, eps=0.01):
    x_true, x_pred, _ = one_hot_encode(x_true, x_pred)
    correct = torch.sum(x_true == x_pred)
    wrong = torch.sum(x_true != x_pred)

    return correct / (correct + wrong)

def psd(y):
    #y = y / (np.max(y, axis=(-1, -2), keepdims=True) + 0.01)
    bins=np.arange(0, 1.0, 0.02) # changed from 0.52
    
    # Compute 2D FFT onp.fft.fft2f the input image
    ffts = np.fft.fft2(y)
    ffts = np.fft.fftshift(ffts)
    ffts = np.abs(ffts) ** 2

    # Compute the frequency grids
    freq = np.fft.fftshift(np.fft.fftfreq(y.shape[-1]))
    freq2 = np.fft.fftshift(np.fft.fftfreq(y.shape[-2]))
    kx, ky = np.meshgrid(freq, freq2)
    kx = kx.T
    ky = ky.T
    # Compute PSD by binning wavenumbers
    x = [
        binned_statistic(
            np.sqrt(kx.ravel() ** 2 + ky.ravel() ** 2),
            values=np.vstack(ffts[i].ravel()).T,
            statistic="mean",
            bins=bins,
        ).statistic
        for i in range(ffts.shape[0])
    ]

    # Compute PSD for the last time step (for normalization)
    x2 = binned_statistic(
        np.sqrt(kx.ravel() ** 2 + ky.ravel() ** 2),
        values=np.vstack(ffts[-1].ravel()).T,
        statistic="mean",
        bins=bins,
    )

    # Normalize the PSD and return it along with bin edges
    return np.array(x)[:, 0, :] / abs(x2.bin_edges[0] - x2.bin_edges[1]), x2.bin_edges

def pearson_coeff(x, y):
    x = torch.flatten(x, start_dim=-2, end_dim=-1)
    y = torch.flatten(y, start_dim=-2, end_dim=-1)   
    
    x = x - x.mean(dim=-1, keepdim=True)
    y = y - y.mean(dim=-1, keepdim=True)    
    
    numerator = torch.sum(x * y, dim=-1)    
    denominator = torch.sqrt(torch.sum(x ** 2, dim=-1) * torch.sum(y ** 2, dim=-1))    
    
    eps = 0.01 
    r = (numerator + eps) / (denominator + eps) 
    return r

def mean_bias(x_true, x_pred):
    bias = x_true - x_pred 
    return torch.mean(bias)

def my_mse(x_true, x_pred):
    return mse_loss(x_true, x_pred)

def from_numpy(x):
    return x.cpu().detach().numpy()
    
def collect_stat(config, func, xhr, pred):
    with torch.no_grad():
        stat = func(xhr.squeeze(), pred.squeeze())
    
    return stat
    
def gather_metrics(config, predictions, eval_dl, funcs): # add predictiosn here 
    stats = {}
    for k, ((xlr, xhr), pred) in enumerate(zip(eval_dl, predictions)):
        #pred = np.expand_dims(pred, axis=1)
        # ignore first dimension, which is number of repeats, get k repeats for a sample 
        #repeats_pred = torch.unsqueeze(predictions[:,k], 1) 
        pred = torch.from_numpy(pred).to(xhr.device)
        for fn, ff in funcs.items():
            stats[fn] = collect_stat(config, ff, xhr, pred) # should compute mean stat if (2, 1, 10, 10) for instance?
    
    return stats

def evaluation_pipeline(config, samplers, dataloaders, coords, from_mem, seg_evl, path='/nesi/project/niwa03712/bailiet/preds'):  
    loc_dict = get_loc_dict()
    mk_dict = lambda: {ln: {} for ln in loc_dict}
    stat_dict, predictions, _psd = mk_dict(), mk_dict(), mk_dict()

    if seg_evl:
        track_stat = config.training.seg_stats
    else:
        track_stat = config.training.track_stat
        
    # collect and save predictions
    for ev_dl, (sn, sampler) in zip(dataloaders, samplers.items()):
        name = f'eval/{sn}'
        
        # check if values were loaded in from memory
        if from_mem:
            preds = sampler.past_preds
        else:
            preds = save_preds(config, ev_dl, sampler, 1, path=path, flag=sn)
        
        # segment preds via location 
        for ln, (llat, llon) in loc_dict.items():
            if (seg_evl and ln != 'New Zealand'):
                continue 
            predictions[ln][sn] = sl_arr(preds, coords, llat, llon)
            set_dl_trns(ev_dl, partial(sl_arr, coords=coords, s0=llat, s1=llon))
            stat_dict[ln][sn] = gather_metrics(config, predictions[ln][sn], ev_dl, track_stat) 
            
            # metrics which need the whole dataset 
            
            sl_gt = sl_arr(ev_dl.dataset.x_data, coords=coords, s0=llat, s1=llon)
            sl_prd = predictions[ln][sn]
            #slq_prd, slq_gt = data_between(sl_gt, 0.9, sl_prd, 1.0)
            sl_gt = sl_gt if type(sl_gt) != torch.Tensor else to_numpy(sl_gt)
            gt_m = np.mean(sl_gt if type(sl_gt) != torch.Tensor else to_numpy(sl_gt), axis=0)
            prd_m = np.mean(sl_prd, axis=0)
            stat_dict[ln][sn]['MSE_T'] = np.mean(np.square(gt_m - prd_m)) 
            stat_dict[ln][sn]['MAE_T'] = np.mean(np.abs(gt_m - pred_m))
            
            if not seg_evl:
                _psd[ln][sn] = psd(predictions[ln][sn])
        set_dl_trns(ev_dl, None) 
        
        nz_pred = predictions['New Zealand'][sn] = preds
        show_samples(config, ev_dl, nz_pred, coords, None, None, nsample=5, flag=name)
        #func_animation(ev_dl, nz_pred, coords, None, None, flag=name)
       
    print('Statistic dictionary:', stat_dict)    
    
    if not seg_evl:
        # give real data 
        ccam = to_numpy(dataloaders[0].dataset.x_data) 
        for ln, (llat, llon) in loc_dict.items(): 
            sl_ccam = sl_arr(ccam, coords, llat, llon)
            predictions[ln]['CCAM'] = sl_ccam
            _psd[ln]['CCAM'] = psd(sl_ccam)

        qq_plot(predictions, loc_dict)
        plot_hist(predictions) 

        # plot psd 
        for ln, smp_psd in _psd.items():
            fig, axs = plt.subplots()
            for sn, (x, y) in smp_psd.items(): 
                plot_psd(axs, x, y, lbl=sn, spt_nm=ln)

            plt.savefig(f'metric_plots/psd_{ln}.pdf')