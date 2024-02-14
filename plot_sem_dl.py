import torch 
import numpy as np
import matplotlib.pyplot as plt
from config import get_my_config
from data import * 
import torch 
from config import get_my_config 
from utils import to_numpy
import matplotlib.pyplot as plt 

config = get_my_config()
data = open_data(config)
multi_sem = torch.load(f'{config.work_dir}/multi_sem0.pth').to(config.device)
#multi_sem.inference_mode = True

dataset = data['train_pri'][-1]
save_dir = '/home/bailiet/DDM/mlde/semantic'

def plot(im, axs):
    im = im[0].squeeze()
    axs.imshow(im, cmap='Blues')

for k, (xlr, gt_sem) in enumerate(dataset):
    xlr = xlr.to(config.device)
    xhr = to_numpy(gt_sem)
    
    pred_sem = to_numpy(multi_sem(xlr))
    xlr = to_numpy(xlr)
    
    if k == 0:
        print(pred_sem[0, 0, :, :].shape)
        plt.figure()
        plt.imshow(pred_sem[0, 0, :, :].squeeze(), cmap='Blues')
        plt.savefig('/home/bailiet/DDM/mlde/semantic/preds_ch0.png')
        #print(pred_sem[0, 1, :, :])
        plt.figure()
        plt.imshow(pred_sem[0, 1, :, :].squeeze(), cmap='Blues')
        plt.savefig('/home/bailiet/DDM/mlde/semantic/preds_ch1.png')
        #print(pred_sem)
        plt.figure()
        plt.imshow(pred_sem[0, 2, :, :].squeeze(), cmap='Blues')
        plt.savefig('/home/bailiet/DDM/mlde/semantic/preds_ch2.png')
        #print(pred_sem)
        plt.figure()
        plt.imshow(pred_sem[0, 3, :, :].squeeze(), cmap='Blues')
        plt.savefig('/home/bailiet/DDM/mlde/semantic/preds_ch3.png')
        plt.close()
    
    fig, axs = plt.subplots(1, 3)
    plot(xlr, axs[0])
    plot(gt_sem, axs[1])
    plot(pred_sem, axs[2])
    plt.savefig(f'{save_dir}/preds{k}')
    
    if k == 5:
        break 