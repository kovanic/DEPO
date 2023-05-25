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
    # val_loss_flow = []
    val_loss_q = []
    val_loss_t = []
    for data in tqdm(val_loader):
        for key in data.keys():
            if key in ('image_0', 'image_1', 'K_0', 'K_1', 'flow_0to1', 'mask'):
                data[key] = data[key].to(device)   
        B = data['image_0'].size(0)
        flow, q, t = model(
            img_q=data['image_0'], img_s=data['image_1'],
            K_q=data['K_0'], K_s=data['K_1'],
            scales_q=0.125 * torch.ones((B, 2), device=device),
            scales_s=0.125 * torch.ones((B, 2), device=device),
            H=60, W=80)
        loss_flow, loss_q, loss_t = val_loss(flow, q, t, data['T_0to1'], None, None)
        # val_loss_flow.append(loss_flow.sum().item())
        val_loss_q.append(loss_q.sum().item())
        val_loss_t.append(loss_t.sum().item())
    return (
            None, # np.sum(val_loss_flow) / len(val_loader.dataset),
            np.sum(val_loss_q) / len(val_loader.dataset),
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
            train_loss_epoch = {'total':0., 'flow': 0., 'q': 0., 't':0.}
            train_batch_loss = {'total':0., 'flow': 0., 'q': 0., 't':0.}
            
            model.train()
            for i, data in tqdm(enumerate(train_loader), total=n_steps_per_epoch):
                for key in data.keys():
                    if key in ('image_0', 'image_1', 'K_0', 'K_1', 'flow_0to1', 'mask'):
                        data[key] = data[key].to(device)
                
                B = data['image_0'].size(0)
                flow, q, t = model(
                    img_q=data['image_0'], img_s=data['image_1'],
                    K_q=data['K_0'], K_s=data['K_1'],
                    scales_q=0.125 * torch.ones((B, 2), device=device),
                    scales_s=0.125 * torch.ones((B, 2), device=device),
                    H=60, W=80)
                
                if hasattr(model, 'loss_weights'):
                    weights = model.pose_regressor.loss_weights
                else:
                    weights = None
    
                loss, flow_loss, q_loss, t_loss = train_loss(flow, q, t, data['T_0to1'], data['flow_0to1'], data['mask'], weights)
                loss = loss / n_accum_steps
                loss.backward()
                
                train_batch_loss['total'] += loss.item() * n_accum_steps * batch_size
                train_batch_loss['flow'] += flow_loss.item() * batch_size
                train_batch_loss['q'] += q_loss.item() * batch_size
                train_batch_loss['t'] += t_loss.item() * batch_size
                
                if ((i + 1) % n_accum_steps == 0) or ((i + 1) == n_steps_per_epoch):
                    # nn.utils.clip_grad_norm_(model.parameters(), 5)
                    
                    optimizer.step()
                    model.zero_grad(set_to_none=True)
                    train_loss_epoch['total'] += train_batch_loss['total']
                    train_loss_epoch['flow'] += train_batch_loss['flow']
                    train_loss_epoch['q'] += train_batch_loss['q']
                    train_loss_epoch['t'] += train_batch_loss['t']
                    
                    wandb.log({
                        "Train batch loss (total)": train_batch_loss['total'] / (batch_size * n_accum_steps),
                        "Train batch loss (flow)": train_batch_loss['flow'] / (batch_size * n_accum_steps),
                        "Train batch loss (q)": train_batch_loss['q'] / (batch_size * n_accum_steps),
                        "Train batch loss (t)": train_batch_loss['t'] / (batch_size * n_accum_steps)
                    })
                    
                    train_batch_loss = {'total':0., 'flow': 0., 'q': 0., 't':0.}
                    scheduler.step()
                    
                #SWA update at each n_steps_between_swa_updates steps of last `n_epochs_swa` epochs.
                if swa and scheduler.swa_needs_update():
                    swa_model.update_parameters(model)
           
            data = None
            for key, val in train_loss_epoch.items():
                train_loss_epoch[key] = val / len(train_loader.dataset)
        
            if (epoch + 1) % repeat_val_epoch == 0:
                loss_val_epoch_flow, loss_val_epoch_q, loss_val_epoch_t = validate(model, val_loss, val_loader, device)

            print(f'epoch {epoch}: val loss(flow)={loss_val_epoch_flow},\n val loss(q)={loss_val_epoch_q}, val loss(t)={loss_val_epoch_t}')

            wandb.log({
                       "Train loss epoch (total)": train_loss_epoch['total'],
                       "Train loss epoch (flow)": train_loss_epoch['flow'],
                       "Train loss epoch (q)": train_loss_epoch['q'],
                       "Train loss epoch (t)": train_loss_epoch['t'],
                       # "Val loss epoch(flow)": loss_val_epoch_flow,
                       "Val loss epoch(q)": loss_val_epoch_q,
                       "Val loss epoch(t)": loss_val_epoch_t
                      })

            if (epoch + 1) % repeat_save_epoch == 0:
                save_model(model, epoch, model_save_path, optimizer, scheduler)
        if swa:
            # I haven't used BatchNorm anywhere
            # update_bn(loader_train, swa_model, device)
            save_model(swa_model.module, epoch, model_save_path+'swa', optimizer, scheduler)



class MixedScheduler:
    def __init__(self, base_scheduler, swa_scheduler,
                 n_epochs, n_epochs_swa, n_steps_per_epoch,
                 n_swa_anneal_steps, n_steps_between_swa_updates):
        self.base_scheduler = base_scheduler
        self.swa_scheduler = swa_scheduler
        self.scheduler = self.base_scheduler
        self.n_epochs = n_epochs
        self.n_epochs_swa = n_epochs_swa
        self.n_steps_per_epoch = n_steps_per_epoch
        self.n_swa_anneal_steps = n_swa_anneal_steps
        self.n_steps_between_swa_updates = n_steps_between_swa_updates
        self._step = 0
        self._swa_step = 0
        self.swa_active = False
        
    def swa_needs_update(self):
        if (self.swa_active and
            (self._swa_step - self.n_swa_anneal_steps > 0) and
            (((self._swa_step - self.n_swa_anneal_steps) % self.n_steps_between_swa_updates == 0) or 
            (self._swa_step == self.n_steps_per_epoch - 1))):
            return True
        else:
            return False
        
    def get_epoch(self):
        return self._step // self.n_steps_per_epoch
    
    def switch_scheduler(self):
        if (self.get_epoch() < self.n_epochs - self.n_epochs_swa):
            self.scheduler = self.base_scheduler
        else:
            self.scheduler = self.swa_scheduler
            self.swa_active = True
            self._swa_step += 1
            
    def step(self):
        self.switch_scheduler()
        self.scheduler.step()
        self._step += 1
        
    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
    
            
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
        for key in data.keys():
            if key in ('image_0', 'image_1', 'K_0', 'K_1', 'flow_0to1', 'mask'):
                data[key] = data[key].to(device)
                             
        B = data['image_0'].size(0)
        flow, q, t = model(
            img_q=data['image_0'], img_s=data['image_1'],
            K_q=data['K_0'], K_s=data['K_1'],
            scales_q=0.125 * torch.ones((B, 2), device=device),
            scales_s=0.125 * torch.ones((B, 2), device=device),
            H=60, W=80)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
 

