import torch
import matplotlib.pyplot as plt

def load_checkpoint(path, device):
    if device == 'cpu':
        checkpoint = torch.load(path, map_location=device)
    else:
        checkpoint = torch.load(path)
    return checkpoint
    

    
def save_model(model, epoch, path, optimizer=None, scheduler=None):
    checkpoint = { 
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'lr_sched': scheduler if scheduler is not None else None
    }
    torch.save(checkpoint, f'{path}_{epoch}.pth')

    
def plot_schedule(scheduler, steps):
    lrs = []
    for _ in range(steps):
        lrs.append(scheduler.get_last_lr())
        scheduler.step()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(lrs)