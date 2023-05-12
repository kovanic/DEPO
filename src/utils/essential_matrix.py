import torch

def pose_from_essential_marix(E):
    U, S, Vt = torch.linalg.svd(E)
    S[:, -1] = 0
    S[:, :2] = (S[:, 0] + S[:, 1])[:, None] / 2.
    
    E = U @ torch.diag_embed(S) @ Vt
    U, S, Vt = torch.linalg.svd(E)

    W = torch.Tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    
    t = U[:, :, -1]
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    return t, R1, R2
    
def make_skew_symmetric_3x3(x: torch.Tensor):
    T_x = torch.zeros((x.size(0), 3, 3), dtype=torch.float32, device=x.device)
    T_x[:, 0, 1] = -x[:, 2]
    T_x[:, 0, 2] = x[:, 1]
    T_x[:, 1, 0] = x[:, 2]
    T_x[:, 1, 2] = -x[:, 0]
    T_x[:, 2, 0] = -x[:, 1]
    T_x[:, 2, 1] = x[:, 0]
    return T_x
        

def essential_matrix_from_T(T: torch.Tensor) -> torch.Tensor:
    '''
    Calculate essential matrix from relative pose.
    Note: essential matrix is defined only up to sign
    '''
    return T[:, :3, :3] @ make_skew_symmetric_3x3(T[:, :3, 3])
    