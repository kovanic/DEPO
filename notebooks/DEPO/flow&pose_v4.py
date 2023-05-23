#!/usr/bin/env python
# coding: utf-8

# 
# Twins-PVT-L + QuadTree(B) + flow & pose (deterministic)

# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[1]:


import sys
sys.path.append('../../src')


# In[2]:


import torch

from torch.utils.data import DataLoader

from data.scannet.utils_scannet_fast import ScanNetDataset
from DEPO.depo import depo_v4
from training.train_depo_pose_and_flow import train
from training.loss_depo import LossMixedDetermininstic
from utils.model import load_checkpoint

from transformers import get_scheduler


# #### Data

# In[3]:


train_data = ScanNetDataset(
    root_dir='/home/project/data/ScanNet/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_train_ft.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=True
)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, drop_last=True, pin_memory=True, num_workers=0)

val_data = ScanNetDataset(
    root_dir='/home/project/data/ScanNet/scans/',
    npz_path='/home/project/code/data/scannet_splits/smart_sample_val.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    calculate_flow=True
)

val_loader = DataLoader(val_data, batch_size=8, shuffle=False, drop_last=False, pin_memory=True, num_workers=0)


# #### Config

# In[4]:


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

config = dict(
    experiment_name='flow_and_pose_v4',
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),

    n_epochs=10,
    n_steps_per_epoch=len(train_loader.dataset) // train_loader.batch_size,
    n_accum_steps=8,
    batch_size=train_loader.batch_size,

    swa=False,
    n_epochs_swa=None,
    n_steps_between_swa_updates=None,

    repeat_val_epoch=1,
    repeat_save_epoch=1,

    model_save_path='../../src/weights/flow_and_pose_v4'
)

config['n_warmup_steps'] = int(config['n_steps_per_epoch'] * 0.5)


# #### Model

# In[6]:


model = depo_v4().to(config['device'])

for name, p in model.named_parameters():
    if 'self_encoder' in name:
        p.requires_grad = False
    else:
        p.requires_grad = True


# #### Loss & Optimizer & Scheduler

# In[7]:


val_loss = LossMixedDetermininstic(mode='val')
train_loss = LossMixedDetermininstic(mode='train')


# In[8]:


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-6)


# In[24]:


scheduler = get_scheduler(
    "cosine",    
    optimizer=optimizer,
    num_warmup_steps=config['n_warmup_steps'],
    num_training_steps=config['n_steps_per_epoch'] * config['n_epochs']
)


# #### Train & val

# In[25]:


train(model, optimizer, scheduler, train_loss, val_loss, train_loader, val_loader, config, **config)

