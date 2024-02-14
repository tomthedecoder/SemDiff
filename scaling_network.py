import torch 
print(torch.cuda.is_available()) 
import torch.nn as nn 
from utils import train_model 
from config import get_my_config 
from data import *
from torch.utils.data import DataLoader
from prior_model import PriorModel
import torch.nn.functional as F
from metrics import evaluation_pipeline
import numpy as np 
from torchvision.models.vgg import vgg16

class ModelWrapper:
    def __init__(self, unet, do_norm=False, past_preds=None, memory_based=False):
        self.unet = unet
        self.past_preds = np.where(past_preds < 0.0005, 0.0, past_preds) if past_preds is not None else None 
        self.memory_based = memory_based 
        self.do_norm = do_norm
        
    def __call__(self, x):
        if not self.memory_based:
            pred = self.unet(x)
        else:
            pred = next(self.past_preds)
            
        if self.do_norm:
            pred /= torch.amax(pred, dim=(-1, -2), keepdim=True)
            
        return torch.where(pred < 0.005, 0.0, pred)
    
    def eval(self):
        self.unet.eval()
        
    def train(self):
        self.unet.train()

def conv_act(ch0, ch1, ks, act=nn.ReLU()):
    return nn.Conv2d(ch0, ch1, ks, padding=1), act

def gpconv_act(ch0, ch1, ks, ngp, act=nn.ReLU()):
    return nn.GroupNorm(ngp, ch0), *conv_act(ch0, ch1, ks, act)

class ResidualBlock(nn.Module):
    def __init__(self, nch=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Sequential(*conv_act(nch, nch, 3))
        self.conv2 = nn.Conv2d(nch, nch, 3, padding=1)
        
    def forward(self, x):
        z = self.conv1(x)
        x = self.conv2(z) + x
        x = self.relu(x)
        
        return x        

class ScaleNetwork(nn.Module):
    def __init__(self, nchnls=32, nlayers=10, nch0=1):
        super().__init__()

        # Initialize layers
        self.init = nn.Sequential(torch.nn.Conv2d(nch0, nchnls, 3, padding=1), torch.nn.BatchNorm2d(nchnls), torch.nn.ReLU())
        self.convN = nn.Conv2d(nchnls, 1, 3, padding=1)
        self.process = nn.Sequential(*[ResidualBlock(nchnls) for _ in range(nlayers)])
        self.celu = nn.CELU()
        self.relu = nn.ReLU()

    def forward(self, x):
        z = self.init(x)
        z = self.process(z)
        x = self.convN(z)
        x = self.celu(x)
        
        return x

def init_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)    
        
def apply_mask(y):
    _max = torch.amax(y, dim=(-1, -2), keepdim=True)
    ys = y / _max 
    mask = torch.where(ys < 0.1, 0.0, 1.0)
    
    return mask * y 
    
class BasicLoss:
    def __init__(self, config):
        vgg = vgg16(pretrained=True)
        self.loss_net = nn.Sequential(*list(vgg.features)[:31])
        self.loss_net = self.loss_net.to(config.device).eval()
        self.mse_loss = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
    
    def __call__(self, x_pred, x_true):
        content = self.l1(x_pred, x_true)
        
        return content
    
def normalise(x):
    max_x = np.max(x, axis=(-1, -2), keepdims=True)
    return x / max_x
    
# build a dataloader from sampler (SDM). Can be from memory. The DL mapping is sampelr HR -> HR (ground truth)
def dl_frm_np(config, dl, nparr=None, sl=None, trn=False, trfns=None, mk90=False):
    x_data = dl.dataset.x_data

    if sl is not None:
        nparr = nparr[sl]
        x_data = x_data[sl]
        input_data = normalise(nparr)
        
        if mk90:
            input_data, x_data = data_between(x_data, 0.9, X=input_data, q_high=1.0)
            
        m2 = np.max(to_numpy(x_data), axis=(-1, -2), keepdims=True)
        
        if trn:
            x_data = m2*input_data
    else:
        m2 = torch.amax(x_data, dim=(-1, -2), keepdim=True)
        input_data = normalise(to_numpy(x_data))
        
    ds = PairedDataset(input_data, x_data)
    dl = DataLoader(ds, batch_size=config.training.batch_size if trn else 1, shuffle=False)
    
    return dl 
    
def concept_condition(model, conc_sets, cond_sets, config, nmeta_epochs=1, nconc_epochs=1, ncond_epochs=1, flag=''):
    train_conc, val_conc = conc_sets
    train_cond, val_cond = cond_sets 
    wrk_dir = config.work_dir
    
    conc_lset, cond_lset = [], []
    lr = 1e-4
    
    for n in range(nmeta_epochs):
        print('Concept step:')
        model, mv1 = train_model(config, model, train_conc, val_conc, num_epochs=nconc_epochs, criterion=BasicLoss(config), val_criterion=BasicLoss(config), flag='conc', optimiser=torch.optim.Adam(model.parameters(), lr=lr), do_save0=True)
        conc_lset.append(mv1[-1])
        
        print('Condition step:')
        model, mv2 = train_model(config, model, train_cond, val_cond, num_epochs=ncond_epochs, criterion=BasicLoss(config), val_criterion=BasicLoss(config), flag=flag, optimiser=torch.optim.Adam(model.parameters(), lr=lr), do_save0=True)
        cond_lset.append(mv2[-1])

    plt.figure()
    print(conc_lset, cond_lset)
    plt.plot(conc_lset, label='concept')
    plt.plot(cond_lset, label='condition')
    plt.legend()
    plt.savefig(f'/home/bailiet/DDM/mlde/slurm/lsets.pdf')
                
    return model 
        

# train a post process pipeline for the pretrained sdm sampler, either from using samples from memory or directly sampling the wrapper class 
def train_post_proc(config, run_cc=False, run_trn=False, krepeats=1, load=False):
    def load(path):
        array = np.load(path)
        shape = array.shape
        array = np.reshape(array, (shape[0], 1, shape[-2], shape[-1]))
        
        return array 
        
    data = open_data(config) 
    path = '/nesi/project/niwa03712/bailiet'
    samples = {
                   'SemSDM2': (load(f'{path}/preds_SemSDM2.npy'), load(f'{path}/preds_SemSDM2_finetune.npy')),
                   'SDM': (load(f'{path}/preds_SDM.npy'), load(f'{path}/preds_SDM_finetune.npy')),
              }
    
    dataloaders, models = [], {}
    
    for n, (sdm_nm, (tsmp, fsmp)) in enumerate(samples.items()):
        print(f'Training SDM instance {sdm_nm}', end='\n')
        
        # construct datasets for pretrain and finetuning 
        trn_dl = dl_frm_np(config, data['train'], trn=True) 
        evl_dl = dl_frm_np(config, data['eval'], trn=False)    
        val_dl = dl_frm_np(config, data['val'], trn=True) 

        sdm_tdl = dl_frm_np(config, data['train'], fsmp, sl=slice(0, 700), trn=True)
        sdm_edl = dl_frm_np(config, data['eval'], tsmp, sl=slice(0, -1), trn=False, mk90=False)
        sdm_vdl = dl_frm_np(config, data['train'], fsmp, sl=slice(700, 1000), trn=True)
        
        # for qq stuff 
        """"if sdm_nm == 'SemSDM2':
            sem_sdm = torch.load(f'{config.work_dir}/_SemSDM2_cc_repeat0')
            sem_sdm = ModelWrapper(sem_sdm)
            models['SemDiff(cc)'] = sem_sdm
            dataloaders.append(sdm_edl)
            
            sem_sdm2 = torch.load(f'{config.work_dir}/SemSDM2_repeat5.pth')
            sem_sdm2 = ModelWrapper(sem_sdm2)
            #models['SemDiff'] = sem_sdm2 
            #dataloaders.append(sdm_edl)
        else:
            scl_sdm = torch.load(f'{config.work_dir}/SDM_repeat9.pth')
            scl_sdm = ModelWrapper(scl_sdm)
            #models['ScaleSDM'] = scl_sdm
            #dataloaders.append(sdm_edl)
            
            scl_sdm2 = torch.load(f'{config.work_dir}/SDM_cc_repeat0')
            scl_sdm2 = ModelWrapper(scl_sdm2)
            #models['ScaleSDM(cc)'] = scl_sdm2 
            #dataloaders.append(sdm_edl)"""
            
        # repeat training for this particular sdm instance 
        for k in range(krepeats):
            if run_cc:
                cc_scalenet = PriorModel(config).to(config.device)
                init_weights(cc_scalenet)
                
                flag = f'{sdm_nm}_cc_repeat{k}'

                if load or True:
                    cc_scalenet = torch.load(f'{config.work_dir}/SDM_cc_repeat0')
                else:
                    cc_scalenet = concept_condition(cc_scalenet, (trn_dl, val_dl), (sdm_tdl, sdm_vdl), config, 
                                                    nmeta_epochs=100, nconc_epochs=1, ncond_epochs=2, flag=flag)                    
                                    
                models[flag] = ModelWrapper(cc_scalenet) 
                dataloaders.append(sdm_edl)
            
            if run_trn:
                scalenet = PriorModel(config).to(config.device)
                init_weights(scalenet)

                flag = f'{sdm_nm}_repeat{k}'
                lr = config.training.lr
                loss = BasicLoss(config)
                optimiser = torch.optim.Adam(scalenet.parameters())
                num_epochs = config.training.num_epochs
                
                if load or True:
                    scalenet = torch.load(f'{config.work_dir}/{flag}.pth')
                else:
                    print('Pretraining')
                    scalenet, _ = train_model(config, scalenet, trn_dl, val_dl, num_epochs=num_epochs, 
                                              criterion=loss, val_criterion=loss, flag=flag, optimiser=optimiser)

                    print('Finetuning')
                    scalenet, _ = train_model(config, scalenet, sdm_tdl, sdm_vdl, num_epochs=num_epochs//2, 
                                              criterion=loss, val_criterion=loss, flag=flag, optimiser=optimiser)
            
                models[flag] = ModelWrapper(scalenet) 
                dataloaders.append(sdm_edl)          
        
    for k in range(10):
        models[f'UNET_{k}'] = ModelWrapper(torch.load(f'{config.work_dir}/unet{k}.pth')) 
        dataloaders.append(data['eval'])
                
    evaluation_pipeline(config, models, dataloaders, data['coords'], False, False) 
    #sdm_90 = dl_frm_np(config, data['eval'], tsmp, sl=slice(0, -1), trn=False, mk90=True)
    #dataloaders = [sdm_90 for _ in range(5)]
    #evaluation_pipeline(config, models, dataloaders, data['coords'], False, False) 
     
config = get_my_config()
train_post_proc(config, run_cc=True, run_trn=True, krepeats=10, load=True)
