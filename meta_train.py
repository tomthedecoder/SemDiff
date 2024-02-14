from src.ml_downscaling_emulator.score_sde_pytorch_hja22.configs.default_xarray_configs import get_config
from src.ml_downscaling_emulator.score_sde_pytorch_hja22.run_lib import train
from src.ml_downscaling_emulator.score_sde_pytorch_hja22.models.ncsnpp import NCSNpp
from src.ml_downscaling_emulator.score_sde_pytorch_hja22.sampling import get_sampling_fn
from src.ml_downscaling_emulator.score_sde_pytorch_hja22.sde_lib import VESDE
from torch.utils.data import DataLoader, Dataset 
from itertools import chain
import xarray as xr 
import cftime
import numpy as np 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import ml_collections 
import torch.nn.functional as F 

from metrics import * 
from config import get_my_config
from data import * 

from prior_model import PriorModel
from prior_loss import PriorLoss
from utils import train_model, save_preds
from prior_model import * 
from plot import plot_train_info 
from functools import partial 

class SampleWrapper:
    def __init__(self, score_model, sampler, past_preds=None, memory_based=False):
        self.sampler = sampler 
        self.score_model = score_model
        self.score_model.eval()
        
        self.past_preds = past_preds if past_preds is not None else None 
        self.memory_based = memory_based    
        
    def __call__(self, x):
        # normalise x 
        m = torch.amax(x, dim=(-1, -2), keepdim=True)
        x = x / m

        if not self.memory_based:
            sample, _ = self.sampler(self.score_model, x)
        else:
            sample = next(self.past_preds)
        
        sample = torch.where(sample < 0.005, 0.0, sample) 
        
        return sample 
    
    def eval(self):
        self.score_model.eval()
    
    def train(self):
        self.sampler.train()

class UNETWrapper:
    def __init__(self, unet, past_preds=None, memory_based=False):
        self.unet = unet
        self.past_preds = np.where(past_preds < 0.0005, 0.0, past_preds) if past_preds is not None else None 
        self.memory_based = memory_based 
        
    def __call__(self, x):
        if not self.memory_based:
            pred = self.unet(x)
        else:
            pred = next(self.past_preds)
            
        return torch.where(pred < 0.005, 0.0, pred)
    
    def eval(self):
        self.unet.eval()
        
    def train(self):
        self.unet.train()
    
def load_model(config, path, nchannel):
    score_model = NCSNpp(config, nchannel).to(config.device)
        
    sde = VESDE()
    shape = (config.training.batch_size, 1, config.data.image_size, config.data.image_size)
    sampler = get_sampling_fn(config, sde, shape, config.sampling.eps)
    
    full_state_dict = torch.load(path)
    model_state_dict = full_state_dict.get('model', full_state_dict)
    score_model.load_state_dict(model_state_dict)
    
    return score_model

def load_prior_model(config, path):
    prior_model = PriorModel(config)
    
    full_state_dict = torch.load(path)
    model_state_dict = full_state_dict.get('model', full_state_dict)
    prior_model.load_state_dict(model_state_dict)
    
    return prior_model

def train_plot(diff_config, my_config, score_model, train_dl, eval_dl, sampler, flag=''):
    train_stats = Statistics('train', my_config.training.track_stat)
    val_stats = Statistics('val', my_config.training.track_stat)
    score_model, statistics = train(diff_config, score_model, my_config.work_dir, train_dl, eval_dl, sampler, train_stats, val_stats, flag=flag, initial_epoch=0)

def get_sampler(diff_config):
    sde = VESDE()
    shape = (diff_config.training.batch_size, 1, diff_config.data.image_size, diff_config.data.image_size)
    sampler = get_sampling_fn(diff_config, sde, shape, diff_config.sampling.eps) 
    
    return sampler 
    
def train_Kmodels(nrepeats=1, model_name='unet', load=False):
    config = get_my_config()
    dls = open_data(config)
    
    dataloaders, models = [], {}
    b = False
    
    # train models
    for n in range(nrepeats): 
        # cases of unet, multi_sem...  
        if model_name == 'unet':
            if load:
                unet = torch.load(f'{config.work_dir}/unet{n}.pth') 
            else:
                unet, _ = train_model(config, PriorModel(config).to(config.device), dls['train'], dls['val'], num_epochs=config.training.num_epochs, criterion=nn.MSELoss(), flag=f'unet{n}') 
                
            model = UNETWrapper(unet)
            dataloaders.append(dls['eval'])
            models[f'unet{n}'] = model 
        elif model_name == 'multi_sem':
            b = True
            multi_sem = PriorModel(config).to(config.device)
            
            def CEL(y_pred, y_target):
                y_target = y_target.squeeze().long()
                if len(y_target.shape) == 2:
                    y_target = y_target.unsqueeze(0)
                return nn.CrossEntropyLoss()(y_pred, y_target)
            
            def val_loss(y_pred, y_target):
                y_pred = torch.argmax(y_pred, dim=1, keepdim=True)
                return F.mse_loss(y_pred, y_target)
            
            multi_sem, _ = train_model(config, multi_sem, dls['train_pri'][-1], dls['val_pri'][-1], num_epochs=config.training.num_epochs, criterion=CEL, val_criterion=F.mse_loss, flag=f'multi_sem{n}') 
            multi_sem.inference_mode = True
            
            dataloaders.append(dls['eval_pri'][-1])
            models[f'multi_sem{n}'] = multi_sem 
        elif model_name == 'cascade':
            b = True
            prior_model = train_prior_model(config, dls['train_pri'], dls['val_pri'], flag=f'sem_repeat{n}') 
            for k in range(prior_model.size()):
                models[f'sem_n{n}_k{k}'] = prior_model.get_sub_chain(k) 
                dataloaders.append(dls['eval_pri'][k]) 
                
    evaluation_pipeline(config, models, dataloaders, dls['coords'], False, b)     
    
def meta_train():
    # config and data collection 
    diff_config = get_config()
    my_config = get_my_config()
    dls = open_data(my_config)
    
    # sampler for diffusion model 
    sampler = get_sampler(diff_config)
    path = '/nesi/project/niwa03712/bailiet'
    from_mem = my_config.eval.from_mem
    model_flag = f''
    
    # added to as model is trained 
    samplers = {} 
    dataloaders = [] 
    
    use_chain = my_config.prior.use_chain
    name = 'AB_' if not use_chain else ''
    
    # add all models in chain
    if my_config.train_prior:
        if my_config.training.load and my_config.prior.cascade:
            samplers, dataloaders = load_cascade(my_config, dls, use_chain, samplers=samplers, dataloaders=dataloaders)
        elif my_config.prior.cascade:
            prior_model = train_prior_model(my_config, dls['train_pri'], dls['val_pri'], flag='sem') 
            for n in range(prior_model.size()):
                samplers[f'{name}sem_{n}'] = prior_model.get_sub_chain(n) 
                dataloaders.append(dls['eval_pri'][n])
        else:
            #prior_model = train_prior_model(my_config, dls['train_pri'], dls['val_pri'], flag='prior1')
            prior_model = torch.load(f'{my_config.work_dir}/prior1.pth')
            samplers['sem'] = prior_model.to(my_config.device) 
            dataloaders.append(dls['eval_pri'][-1])
    
    # UNET baseline 
    if my_config.train_unet:
        unet = torch.load('/nesi/project/niwa03712/bailiet/add_work/unet.pth') if my_config.training.load else PriorModel(my_config).to(my_config.device)
        _, unet = train_model(my_config, unet, dls['train'], dls['val'], num_epochs=my_config.training.num_epochs, criterion=nn.MSELoss(), flag='unet') 
        samplers['unet'] = UNETWrapper(unet) 
        dataloaders.append(dls['eval'])
    else:
        unet = torch.load('/nesi/project/niwa03712/bailiet/add_work/unet.pth')
    
    # SDM diffusion model 
    if my_config.train_diff:
        all_model = load_model(diff_config, f'{my_config.work_dir}/checkpoints/all.pth', 2) if my_config.training.load else NCSNpp(diff_config, 2).to(diff_config.device)
        train_plot(diff_config, my_config, all_model, dls['train'], dls['val'], sampler, flag='all')
    else:
        all_model = load_model(diff_config, f'{my_config.work_dir}/checkpoints/all.pth', 2)
    #samplers[f'{model_flag}SDM_finetune'] = SampleWrapper(all_model.to(my_config.device), sampler)
    #dataloaders.append(dls['train'])
    
    #samplers[f'{model_flag}UNET'] = UNETWrapper(unet.to(my_config.device)) 
    #dataloaders.append(dls['eval'])
    
    # Sem SDM model 
    lvl = len(my_config.prior.taus) - 1
    if my_config.train_sem_diff:
        sem_model = load_model(diff_config, f'{my_config.work_dir}/checkpoints/sem_lvl{lvl}.pth', 3) if my_config.training.load else NCSNpp(diff_config, 3).to(diff_config.device)
        train_plot(diff_config, my_config, sem_model, dls['train_sdiff'], dls['eval_sdiff'], sampler, flag=f'sem_lvl{lvl}_2.pth')
    else:
        sem_model = load_model(diff_config, f'{my_config.work_dir}/checkpoints/sem_lvl{lvl}.pth', 3)   
    #samplers[f'{model_flag}SemSDM2_finetune'] = SampleWrapper(sem_model.to(my_config.device), sampler) 
    #dataloaders.append(dls['train_sdiff'])

    evaluation_pipeline(my_config, samplers, dataloaders, dls['coords'], False, my_config.train_prior) 
    
#meta_train()
train_Kmodels(nrepeats=1, model_name='multi_sem', load=False)