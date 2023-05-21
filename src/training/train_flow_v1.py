import sys
sys.path.append('../src')

import torch
from torch.optim.swa_utils import AveragedModel

import numpy as np
from tqdm.auto import tqdm
import wandb

from utils.model import save_model
from scipy.spatial.transform import Rotation



@torch.no_grad()
def validate(model, val_loss, val_loader, device):
    model.eval()
    val_loss_q = []
    val_loss_t = []
    for data in tqdm(val_loader):
        data['flow'] = data['flow'].to(device) 
        data['K_q'] = data['K_q'].to(device)
        data['K_s'] = data['K_s'].to(device)
        q, t, m = model(
                data['flow'], data['K_q'], data['K_s'], (0.25, 0.25), (0.25, 0.25)
               )
        loss_t, loss_q = val_loss(q, t, m, data['rel_pose'])
        val_loss_q.append(loss_q.sum().item())
        val_loss_t.append(loss_t.sum().item())
        
    del data
    return (np.sum(val_loss_q) / len(val_loader.dataset),
            np.sum(val_loss_t) / len(val_loader.dataset))


@torch.no_grad()
def test(model, loader, device):
    model.eval()
    results = {'pair_id': [], 
               't_gt': [],
               'R_gt': [],
               't': [],
               'R': []}
    for data in tqdm(loader):
        data['flow'] = data['flow'].to(device) 
        data['K_q'] = data['K_q'].to(device)
        data['K_s'] = data['K_s'].to(device)
        q, t, m = model(
                data['flow'], data['K_q'], data['K_s'], (0.25, 0.25), (0.25, 0.25)
               )
        t = t * m
        q = q.cpu().numpy()
        t = t.cpu().numpy()
        #[w, x, y, z] -> [x, y, z, w]
        R = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        T_gt = data['rel_pose'].numpy()
        t_gt = T_gt[:, :3, 3]
        R_gt = T_gt[:, :3, :3]
        
        results['pair_id'].append(data['pair_id'])
        results['t_gt'].append(t_gt)
        results['t'].append(t)
        results['R_gt'].append(R_gt)
        results['R'].append(R)
        
    for key, val in results.items():
        results[key] = np.concatenate(val)
    return results


def train(model, optimizer, scheduler, train_loss, val_loss,
          train_loader, val_loader,
          config, experiment_name, device,
          n_epochs, n_steps_per_epoch, n_accum_steps,
          swa, n_epochs_swa, n_steps_between_swa_updates,
          repeat_val_epoch, repeat_save_epoch,
          batch_size, model_save_path, **args):
    
    with wandb.init(project="Diploma", config=config, name=experiment_name) as exp:
        if swa:
            swa_model = AveragedModel(model)
            swa_model.to(device)
        
        for epoch in range(n_epochs):
            train_loss_epoch = 0
            train_batch_loss = 0
            
            model.train()
            for i, data in tqdm(enumerate(train_loader), total=n_steps_per_epoch):
                data['flow'] = data['flow'].to(device) 
                data['K_q'] = data['K_q'].to(device)
                data['K_s'] = data['K_s'].to(device)
                q, t, m = model(
                        data['flow'], data['K_q'], data['K_s'], (0.25, 0.25), (0.25, 0.25)
                       )
               
                loss_ = train_loss(q, t, m, data['rel_pose']) / n_accum_steps
                loss_.backward()
                
                train_batch_loss += loss_.item() * n_accum_steps * batch_size
                
                if ((i + 1) % n_accum_steps == 0) or ((i + 1) == n_steps_per_epoch):
                    # nn.utils.clip_grad_norm_(model.parameters(), 5)
                    
                    optimizer.step()
                    model.zero_grad(set_to_none=True)
                    train_loss_epoch += train_batch_loss
                    
                    wandb.log({
                        "Train batch loss": train_batch_loss / (batch_size * n_accum_steps)
                    })
                    
                    train_batch_loss = 0
                    
                #SWA update at each n_steps_between_swa_updates steps of last `n_epochs_swa` epochs.
                if (swa and (scheduler.step_ > (n_epochs - n_epochs_swa) * n_steps_per_epoch) and
                   ((scheduler.step_ % n_steps_between_swa_updates == 0) or (scheduler.step_ % step_per_epoch == 0))):
                    swa_model.update_parameters(model)
           
            scheduler.step()
            data = None
            train_loss_epoch /= len(train_loader.dataset)
        
            if (epoch + 1) % repeat_val_epoch == 0:
                loss_val_epoch_q, loss_val_epoch_t = validate(model, val_loss, val_loader, device)

            print(f'epoch {epoch}: train loss={train_loss_epoch}, val loss(q)={loss_val_epoch_q}, val loss(t)={loss_val_epoch_t}')

            wandb.log({
                       "Train loss epoch": train_loss_epoch,
                       "Val loss epoch(q)": loss_val_epoch_q,
                       "Val loss epoch(t)": loss_val_epoch_t
                      })

            if (epoch + 1) % repeat_save_epoch == 0:
                save_model(model, epoch, model_save_path, optimizer, scheduler)
        if swa:
            update_bn(loader_train, swa_model, device)
            save_model(swa_model.module, f'{model_save_path}_swa.pth')

            
@torch.no_grad()
def update_bn(loader, model, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    for data in loader:
        data['flow'] = data['flow'].to(device) 
        data['K_q'] = data['K_q'].to(device)
        data['K_s'] = data['K_s'].to(device)
        model(
            data['flow'], data['K_q'], data['K_s'], (0.25, 0.25), (0.25, 0.25)
            )

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
 

