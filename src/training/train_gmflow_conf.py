import torch
from torch.optim.swa_utils import AveragedModel

import numpy as np
from tqdm.auto import tqdm
import wandb


@torch.no_grad()
def validate(model, loss, val_loader, device):
    model.eval()
    val_loss = []
    for data in tqdm(val_loader):
        B = data['image0'].shape[0]
        data = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.Tensor)}
        preds = model(
                data['image0'],
                data['image1'],
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1]
               )
        loss_ = loss(preds, data['flow_0to1'], data['mask']).item() * B
        val_loss.append(loss_)
    del data
    return np.sum(val_loss) / len(val_loader.dataset)

    
def train(model, optimizer, scheduler, loss, val_loss,
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
                data = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.Tensor)}
                
                preds = model(
                    data['image0'],
                    data['image1'],
                    attn_splits_list=[2, 8],
                    corr_radius_list=[-1, 4],
                    prop_radius_list=[-1, 1]
                   )
               
                loss_ = loss(preds, data['flow_0to1'], data['mask']) / n_accum_steps
                loss_.backward()
                
                train_batch_loss += loss_.item() * n_accum_steps * batch_size
                
                if ((i + 1) % n_accum_steps == 0) or ((i + 1) == n_steps_per_epoch):
                    # nn.utils.clip_grad_norm_(model.parameters(), 5)
                    
                    optimizer.step()
                    scheduler.step()
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
                
            data = None
            train_loss_epoch /= len(train_loader.dataset)
        
            if (epoch + 1) % repeat_val_epoch == 0:
                loss_val_epoch = validate(model, val_loss, val_loader, device)

            print(f'epoch {epoch}: train loss={train_loss_epoch}, val loss={loss_val_epoch}')

            wandb.log({
                       "Train loss epoch": train_loss_epoch,
                       "Val loss epoch": loss_val_epoch
                      })

            if (epoch + 1) % repeat_save_epoch == 0:
                model_save(model, f'{model_save_path}_{epoch}.pth')
        
        if swa:
            update_bn(loader_train, swa_model, device)
            model_save(swa_model.module, f'{model_save_path}_swa.pth')

            
            
            
def model_save(model, name):
    torch.save(model.state_dict(), name)
            
        
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
        data = {k: v.to(device) for k,v in data.items() if isinstance(v, torch.Tensor)}
        # if isinstance(input, (list, tuple)):
        #     input = input[0]
        # if device is not None:
        #     input = input.to(device)

        model(
            data['image0'],
            data['image1'],
            attn_splits_list=[2, 8],
            corr_radius_list=[-1, 4],
            prop_radius_list=[-1, 1]
           )

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)
 

class CustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, n_steps_per_epoch, warmup_epochs=1, warmup_lr_init=1e-6,
                 min_lr=1e-6, last_epoch=-1, verbose=False):

        self.warmup_steps = warmup_epochs * n_steps_per_epoch
        self.warmup_lr_init = warmup_lr_init
        self.min_lr = min_lr
        self._step = -1
        
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        self._step+=1
        if self._step < self.warmup_steps:
            return [self._step / self.warmup_steps * (self.optimizer.param_groups[0]['lr'] - self.min_lr) + self.min_lr, self.optimizer.param_groups[1]['lr']]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]