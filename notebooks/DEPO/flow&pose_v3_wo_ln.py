#!/usr/bin/env python
# coding: utf-8

# FocalNet-B + QuadTree-B + flow & pose (deterministic)

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[2]:


import sys
sys.path.append('../../src')


# In[3]:


import torch

from torch.utils.data import DataLoader

from data.scannet.utils_scannet_fast import ScanNetDataset
from DEPO.depo import depo_v3_wo_ln
from training.train_depo_pose_and_flow import train, validate
from training.loss_depo import LossMixedDetermininstic
from utils.model import load_checkpoint
import numpy as np

from transformers import get_scheduler


# #### Data

# In[4]:


train_data = ScanNetDataset(
    root_dir='/home/project/data/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_train_ft.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=True
)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True, pin_memory=True, num_workers=4)

val_data = ScanNetDataset(
    root_dir='/home/project/data/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_val.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=False
)

val_loader = DataLoader(val_data, batch_size=32, shuffle=False, drop_last=False, pin_memory=True, num_workers=4)


# #### Config

# In[5]:


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = dict(
    experiment_name='flow_and_pose_v3_wo_ln',
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
    n_epochs=10,
    n_accum_steps=4,
    batch_size=train_loader.batch_size,
    n_steps_per_epoch=len(train_loader.dataset) // train_loader.batch_size,
    swa=False,
    n_epochs_swa=None,
    n_steps_between_swa_updates=None,
    repeat_val_epoch=1,
    repeat_save_epoch=1,
    model_save_path='../../src/weights/flow_and_pose_v3_wo_ln'
)

config['n_effective_steps_per_epoch'] = np.ceil(len(train_loader.dataset) / (train_loader.batch_size * config['n_accum_steps'])) 
config['n_warmup_steps'] = config['n_effective_steps_per_epoch'] * 2
config['n_training_steps'] = int(config['n_effective_steps_per_epoch'] * config['n_epochs'])


# #### Model

# In[6]:


model = depo_v3_wo_ln().to(config['device'])

for name, p in model.named_parameters():
    if 'self_encoder' in name:
        p.requires_grad = False
    else:
        p.requires_grad = True


# #### Loss & Optimizer & Scheduler

# In[7]:


val_loss = LossMixedDetermininstic(mode='val')
train_loss = LossMixedDetermininstic(mode='train', add_l2=False)


# In[8]:


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)


# In[9]:


scheduler = get_scheduler(
    "cosine",    
    optimizer=optimizer,
    num_warmup_steps=config['n_warmup_steps'],
    num_training_steps=config['n_training_steps'] 
)


# #### Train & val

# In[ ]:


train(model, optimizer, scheduler, train_loss, val_loss, train_loader, val_loader, config, **config)

