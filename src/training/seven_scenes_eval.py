import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    results = {'pair_id': [], 
               'scene_name': [],
               't_gt': [],
               'R_gt': [],
               't': [],
               'R': []}
    for data in tqdm(loader):
        for key in data.keys():
            if key in ('image_0', 'image_1', 'K_0', 'K_1'):
                data[key] = data[key].to(device)
                
        B = data['image_0'].size(0)
        _, q, t = model(
            img_q=data['image_0'], img_s=data['image_1'],
            K_q=data['K_0'], K_s=data['K_1'],
            scales_q=0.125 * torch.ones((B, 2), device=device),
            scales_s=0.125 * torch.ones((B, 2), device=device),
            H=60, W=80)
        
        q = q.cpu().numpy()
        t = t.cpu().numpy()
        #[w, x, y, z] -> [x, y, z, w]
        R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        T_gt = data['T_0to1'].numpy()
        t_gt = T_gt[:, :3, 3]
        R_gt = T_gt[:, :3, :3]
        
        results['pair_id'].append(data['pair_id'])
        results['scene_name'].append(data['scene_name'])
        results['t_gt'].append(t_gt)
        results['t'].append(t)
        results['R_gt'].append(R_gt)
        results['R'].append(R)
        
    for key, val in results.items():
        results[key] = np.concatenate(val)
    return results

