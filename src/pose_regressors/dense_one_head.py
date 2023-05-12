import torch
from torch import nn
import torch.nn.functional as F
from copy import copy

def create_normalized_grid(image_size=(120, 160)):
    '''Given image size, return grid for pixels positions, normalized to [-0.5, 0.5]. '''
    H, W = image_size
    grid = torch.meshgrid((torch.arange(H), torch.arange(W)), indexing='ij')
    grid = torch.cat((grid[1].unsqueeze(0), grid[0].unsqueeze(0)), dim=0).float()
    grid[1] = grid[1] / (H - 1) - 0.5
    grid[0] = grid[0] / (W - 1) - 0.5
    return grid.unsqueeze(0) # 1 x 2 x H x W


class DensePoseRegressorOneHead(nn.Module):
    def __init__(self, image_size, in_ch, init_loss_weights=[0., -3.]):
        super(DensePoseRegressorOneHead, self).__init__()

        grid = create_normalized_grid(image_size=image_size)
        self.register_buffer('pe', grid)

        loss_weights = torch.nn.Parameter(torch.tensor(init_loss_weights))
        self.register_parameter('loss_weights', loss_weights)

        self.decoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=1, padding=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 5, stride=5),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, (1, 3), 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, (1, 3), 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 7, 1, 1, 0)
        )
        
    def forward(self, x):
        # x: B x 3 x H x W, 3 channels are (u, v confidence)
        B, _, H, W = x.size()
        x = torch.cat((x, self.pe.repeat(B, 1, 1, 1)), dim=1) #B x 5 x H x W
        x = self.decoder(x)

        t = x[:, :3].squeeze()
        q = x[:, 3:] #B x 4 x 1 x 1

        q[:, 0] = torch.abs(q.clone()[:, 0])
        q = q / (q ** 2).sum(dim=1).sqrt()[:, None]
        q = q.squeeze()
        return q, t