import ml_collections
import torch 
import torchmetrics 
import pytorch_ssim  
import numpy as np 
from metrics import * 
from torchmetrics.functional import dice


def get_my_config():
    config = ml_collections.ConfigDict()
    
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    config.ngpus = 1
    config.work_dir = '/nesi/project/niwa03712/bailiet/add_work'
    config.train_prior = False
    config.train_sem_diff = False
    config.train_diff = False   
    config.train_seg = False 
    config.train_unet = False 
    
    config.training = training = ml_collections.ConfigDict()
    training.load = False
    training.save = True
    training.num_epochs = 100
    training.batch_size = 32
    training.lr = 1e-3
    training.gen_lr = 1e-3
    training.dis_lr = 1e-4
    training.log_interval = 10 # for training anything but diffusion models 
    training.track_stat = {'MSE': my_mse, 'pearson': pearson_coeff, 'mbias': mean_bias, 'MAE': torch.nn.L1Loss(), 'psnr': psnr, 'ssim': my_ssim}
    training.seg_stats = {'MSE': my_mse, 'iou': iou, 'pixel accuracy': pixel_accuracy, 'dice score': dice_score}
    
    config.prior = prior = ml_collections.ConfigDict() 
    prior.nch = [32*(i+1) for i in range(3)]
    prior.taus = [0.1, 0.4, 0.6, 1.0] 
    prior.expectation = True
    prior.multi_class = True
    prior.cascade = False  
    prior.use_chain = False
    
    config.fusion = fusion = ml_collections.ConfigDict()
    fusion.num_peaks = 1 
    fusion.min_dist = 30
    fusion.ws = 48

    config.data = data = ml_collections.ConfigDict()
    data.nclasses = 6
    data.coarsen = True
    data.coarse_factor = 8
    data.shape = (168, 168)
    data.nsamples = -1
    data.ptrain = 0.90
    data.peval = 0.05
    data.ninput_ch = 1 
    data.noutput_ch = 1
    
    #data.file_path = "/nesi/nobackup/niwa03712/shared_paths/high-res-data/ACCESS-CM2_ssp370_precip.nc"
    data.file_path = "/nesi/nobackup/niwa03712/shared_paths/high-res-data/ACCESS-CM2_historical_precip.nc"
    
    
    config.boosting = boosting = ml_collections.ConfigDict()
    boosting.nmodels = 4
    
    config.sample = sample = ml_collections.ConfigDict()
    sample.nr = 3
    sample.ns = 3 
    
    config.eval = eval = ml_collections.ConfigDict()
    eval.from_mem = False 
    
    
    return config 
