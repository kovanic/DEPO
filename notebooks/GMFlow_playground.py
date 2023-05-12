#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[17]:


# import gdown
# gdown.download(id='1d5C5cgHIxWGsFR1vYs5XrQbbUiZl9TX2') #gmflow weights


# In[2]:


import sys
sys.path.append('../src')


# In[3]:


import torch

from data.scannet.utils import ScanNetDataset
from matching.gmflow.gmflow.gmflow import GMFlow


# In[6]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GMFlow(num_scales=2, upsample_factor=4, fine_tuning=True)
model.to(device)
model.load_state_dict(torch.load('../src/matching/gmflow/weights/pretrained/gmflow_with_refine_kitti-8d3b9786.pth')['model'])


# In[7]:


train_data = ScanNetDataset(
    root_dir='/home/project/data/scans/',
    npz_path='/home/project/ScanNet/train_indicies_subset.npz',
    intrinsics_path='/home/project/ScanNet/scannet_indices/intrinsics.npz',
    mode='train'
)


# In[10]:


data = train_data[0]
image0 = data['image0'].to(device)
image1 = data['image1'].to(device)


# In[11]:


# model.eval()

with torch.no_grad():
    out = model(image0, image1,
          attn_splits_list=[2, 8],
          corr_radius_list=[-1, 4],
          prop_radius_list=[-1, 1])


# In[18]:


x = {1:2}
z = x.pop(1)
z


# In[101]:


import matplotlib.pyplot as plt
plt.imshow(out[0, 1300].reshape(60, 80).numpy())


# In[86]:


out[0, 4]


# In[39]:


x = torch.rand(2, 2, 60, 80)


# In[40]:


import torch.nn.functional as F
up_flow = F.unfold(4 * x, [3, 3], padding=1)


# In[63]:


model.training


# In[17]:


import matplotlib.pyplot as plt


# In[58]:


out['flow_preds']хъ


# In[55]:


magnitude = (
    (out['flow_preds'][0].squeeze()[0].numpy() / 480 ** 2)
    + (out['flow_preds'][0].squeeze()[1].numpy() / 640 ** 2)) ** 0.5

plt.imshow(magnitude)

