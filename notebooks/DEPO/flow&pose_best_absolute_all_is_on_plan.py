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
from torch.optim.swa_utils import SWALR

from data.scannet.utils_scannet_fast import ScanNetDataset
from DEPO.depo import depo_best
from training.train_depo_pose_and_flow_weighted import train, validate, MixedScheduler
from training.loss_depo import LossMixedDetermininsticWeighted

from utils.model import load_checkpoint, plot_schedule
import numpy as np

from transformers import get_scheduler


# #### Data

# In[4]:


train_data = ScanNetDataset(
    root_dir='/home/project/data/ScanNet/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_train.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=True
)

train_loader = DataLoader(train_data, batch_size=3, shuffle=True, drop_last=True, pin_memory=True, num_workers=3)

val_data = ScanNetDataset(
    root_dir='/home/project/data/ScanNet/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_val.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=False
)

val_loader = DataLoader(val_data, batch_size=3, shuffle=False, drop_last=False, pin_memory=True, num_workers=3)


# #### Config

# In[5]:


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = dict(
    experiment_name='flow_and_pose_best_abs',
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    n_epochs=2,
    n_accum_steps=22,
    batch_size=train_loader.batch_size,
    n_steps_per_epoch=len(train_loader.dataset) // train_loader.batch_size,
    swa=True,
    swa_lr=1e-4,
    n_epochs_swa=1,
    repeat_val_epoch=1,
    repeat_save_epoch=1,
    model_save_path='../../src/weights/flow_and_pose_best_abs'
)

config['n_effective_steps_per_epoch'] = np.ceil(len(train_loader.dataset) / (train_loader.batch_size * config['n_accum_steps'])) 
config['n_warmup_steps'] = int(config['n_effective_steps_per_epoch'] * 0.1)
config['n_training_steps'] = int(config['n_effective_steps_per_epoch'] * (config['n_epochs'] - config['n_epochs_swa']))
config['n_swa_anneal_steps'] = int(config['n_effective_steps_per_epoch'] * 0.1)
config['n_steps_between_swa_updates'] = (config['n_effective_steps_per_epoch'] - config['n_swa_anneal_steps']) // 20


# #### Model

# In[6]:


checkpoint = load_checkpoint(
    '/home/project/code/src/weights/flow_and_pose_best_relative_2.pth',
    config['device'])

model = depo_best()
model.load_state_dict(checkpoint['model'])
model.to(config['device']);


# #### Loss & Optimizer & Scheduler

# In[7]:


val_loss = LossMixedDetermininsticWeighted(mode='val', weights=None)
train_loss = LossMixedDetermininsticWeighted(mode='train', weights=[0., -4., 0.], add_l2=False)


# In[8]:


opt_parameters = []
for name, module in model.named_parameters():
    if 'self_encoder' in name:
        lr = 5e-5
    else:
        lr = 5e-4
    opt_parameters.append({
        'params': module,
        'weight_decay': 0.0 if ('bias' in name) else 1e-6,
        'lr': lr
    })

optimizer = torch.optim.AdamW(opt_parameters)
weights_optimizer = torch.optim.SGD([train_loss.weights], lr=1e-4)


# In[9]:


base_scheduler = get_scheduler(
    "cosine",    
    optimizer=optimizer,
    num_warmup_steps=config['n_warmup_steps'],
    num_training_steps=config['n_training_steps'])

swa_scheduler = SWALR(
    optimizer,
    swa_lr=config['swa_lr'],
    anneal_epochs=config['n_swa_anneal_steps'])

scheduler = MixedScheduler(
    base_scheduler,
    swa_scheduler,
    n_epochs=config['n_epochs'],
    n_epochs_swa=config['n_epochs_swa'],
    n_steps_per_epoch=config['n_effective_steps_per_epoch'],
    n_swa_anneal_steps=config['n_swa_anneal_steps'],
    n_steps_between_swa_updates=config['n_steps_between_swa_updates']
)


# In[10]:


# for step in range(int(config['n_training_steps'] + config['n_effective_steps_per_epoch'])):
#     scheduler.step()
#     if scheduler.swa_needs_update():
#         print(step)


# In[11]:


# import matplotlib.pyplot as plt
# plot_schedule(scheduler, int(config['n_effective_steps_per_epoch'] * config['n_epochs']))


# #### Train & val

# In[ ]:


train(model, optimizer, weights_optimizer, scheduler, train_loss, val_loss, train_loader, val_loader, config, **config)

