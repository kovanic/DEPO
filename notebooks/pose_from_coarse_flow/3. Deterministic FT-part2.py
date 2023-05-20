#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


import sys
sys.path.append('../../src')


# In[3]:


import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from data.scannet.utils_scannet import ScanNetDataset
from matching.gmflow_dense.gmflow_dense import GMflowDensePoseDeterministic
from flow_regressors.regressors import DensePoseRegressorV7

from training.loss_pose import LossPose
from training.loss_deterministic import DeterministicLossMixed
from training.train_deterministic import train, validate
from utils.model import load_checkpoint

from tqdm.auto import tqdm
import wandb
import re


# ### 1. Data

# In[4]:


train_data = ScanNetDataset(
    root_dir='/home/project/data/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_train_ft.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=True
)

train_loader = DataLoader(train_data, batch_size=2, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

val_data = ScanNetDataset(
    root_dir='/home/project/data/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_val.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=True
)

val_loader = DataLoader(val_data, batch_size=2, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)


# ### 2. Configuration

# In[5]:


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = dict(
    general = dict(
        experiment_name='DETERMINISTIC_FT_part2',
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        
        n_epochs=5,
        n_steps_per_epoch=len(train_loader.dataset)//train_loader.batch_size,
        n_accum_steps=16,
        batch_size=train_loader.batch_size,
        
        swa=False,
        n_epochs_swa=None,
        n_steps_between_swa_updates=None,

        repeat_val_epoch=1,
        repeat_save_epoch=1,
          
        model_save_path='../../src/weights/DETERMINISTIC_FT_part2'
    )
)


config['general']['n_warmup_steps'] = int(config['general']['n_steps_per_epoch'] * 0.5)


# ### 3. Model

# In[6]:


checkpoint = load_checkpoint('/home/project/code/src/weights/DETERMINISTIC_FT_4.pth',
                             config['general']['device'])


# In[7]:


regressor = DensePoseRegressorV7(init_loss_weights=[0.0, -3.0, -3.0])
model = GMflowDensePoseDeterministic(regressor, fine_tuning=False)
model.load_state_dict(checkpoint['model'])

model.to(torch.float32)
model.to(config['general']['device']);


# In[8]:


for name, p in model.named_parameters():
    if 'backbone' not in name:
        p.requires_grad = True
    else:
        p.requires_grad = False


# ### 4. Loss, optimizer, scheduler

# In[9]:


train_loss = DeterministicLossMixed()
val_loss = LossPose()


# In[10]:


max_lr = 1e-4
factor = 0.8
weight_decay = 1e-6
n_layers = 6
no_decay = "bias"

pattern = re.compile('layers.[0-9]+')
opt_parameters = []
for name, module in model.named_parameters():
    layer = pattern.findall(name)
    
    if layer:
        layer_n = int(layer[0].split('.')[1]) + 1
        lr = max_lr * factor ** (n_layers - layer_n)
    else:
        lr = max_lr
         
    opt_parameters.append({
        'params': module,
        'weight_decay': 0.0 if (no_decay in name) else weight_decay,
        'lr': lr
    })
    

optimizer = torch.optim.AdamW(opt_parameters)


# In[11]:


scheduler = get_scheduler(
    "cosine",    
    optimizer=optimizer,
    num_warmup_steps=config['general']['n_warmup_steps'],
    num_training_steps=config['general']['n_steps_per_epoch'] * config['general']['n_epochs']
)


# ### 6. Experiment

# In[12]:


train(model, optimizer, scheduler, train_loss, val_loss, train_loader, val_loader, config, **config['general'])

