import torch
from torch import nn
import torch.nn.functional as F

class EssRegressor(nn.Module):
    def __init__(self, n_samples: int=10000):
        super(EssRegressor, self).__init__()
        self.n_samples = n_samples
        
    
    def sample_flow(self, flow: torch.Tensor):
        '''
        Sample N points (the same across batch).
        '''
        B, C, H, W = flow.size()
        mask = torch.rand(1, 1, H, W)
        mask = mask < (self.n_samples / H / W )
        return flow.view(B, C, H*W)[:, :, mask.view(H*W)].permute(0, 2, 1)
         
        
    def normalize_coordinates(self, uv, K):
        '''
        Apply normalization with intrinsics to coordinates in pixels.
        :param uv: coordinates in pixels, B x N x 2
        :param K: intrinsics, B x 3 x 3
        '''
        B, N, _ = uv.size()
        uv_h = torch.cat([uv, torch.ones(B, N, 1, device=uv.device)], dim=2)
        uv_n = uv_h @ torch.linalg.inv(K).transpose(2, 1)
        return uv_n[..., :-1] / uv_n[..., -1].unsqueeze(-1)
    
    
    def reorganize_flow(self, flow: torch.Tensor):
        '''
        [dw, dh, confidence] -> [us, vs, uq, vq, confidence]
        '''
        B, _, H, W = flow.size()
        w, h = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
        coo = torch.cat([w[None, ...], h[None, ...]])[None, ...].to(flow.device) # 1 x 2 x H x W
        return (torch.cat([coo + flow[:, :2, ... ], coo.repeat((B, 1, 1, 1)), flow[:, 2, ...].unsqueeze(1)], dim=1))

   
    def forward(self, flow: torch.Tensor, K_q: torch.Tensor, K_s: torch.Tensor):
        '''
        :param flow: B x 3 x H x W tensor where 3 channels are [dw, dh, confidence]
        :param K_{s, q}: intrinsics, B x 3 x 3
        :return: essential matrix, B x 3 x 3
        '''
        flow = self.reorganize_flow(flow) # B x 5 x H x W
        X = self.sample_flow(flow) # B x n_samples x 5
        B, N, _ = X.size()
        
        X[..., :2] = self.normalize_coordinates(X[..., :2], K_s)
        X[..., 2:4] = self.normalize_coordinates(X[..., 2:4], K_q)

        W = X[..., 4][..., None] # B x n_samples x 1
        
        X = torch.cat([
            (X[..., 0] * X[..., 2])[..., None], (X[..., 0] * X[..., 3])[..., None], X[..., 0][..., None],
            (X[..., 1] * X[..., 2])[..., None], (X[..., 1] * X[..., 3])[..., None], X[..., 1][..., None],
             X[..., 2][..., None], X[..., 3][..., None], torch.ones((B, N, 1), device=X.device)
        ], dim=2)
                
        A = X.transpose(2, 1) @ (W * X)
        _, eig_vectors = torch.linalg.eigh(A)

        E = eig_vectors[:, 0].reshape(B, 3, 3)
        return E
