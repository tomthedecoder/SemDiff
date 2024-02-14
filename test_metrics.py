import numpy as np 
import torch 
from data import open_data 
from config import get_my_config 
from utils import *

config = get_my_config()
data = open_data(config)
data_dir = '/nesi/project/niwa03712/bailiet'

eval_set = data['eval'].dataset.x_data 
semdiff = np.load(f'{data_dir}/preds__SemSDM2_cc_repeat0.npy')

print('MBIAS', np.mean(np.abs(eval_set - semdiff)))