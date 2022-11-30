import torch
from torch import nn


def get_loss_func(cfg):
    if cfg.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif cfg.loss == 'rmse':
        loss_fn = RMSELoss()
    elif cfg.loss == 'l1':
        loss_fn = nn.SmoothL1Loss(reduction='mean', beta=cfg.beta_L1)
    elif cfg.loss == 'double':
        loss_fn = DoubleLoss(w_mse=cfg.w_mse, w_l1=cfg.w_l1, beta_l1=cfg.beta_L1)
    elif cfg.loss == 'huber':
        loss_fn = nn.HuberLoss(reduction='mean', delta=cfg.delta_Huber)

    else:
        raise ValueError('Error in "get_loss_func" function:',
                         f'Wrong loss name. Choose one from ["mse", "rmse", "l1", "double"] ')

    return loss_fn


class DoubleLoss(nn.Module):
    def __init__(self, w_mse=0.5, w_l1=0.5, beta_l1=0.125, eps=1e-9):
        super().__init__()

        self.w_mse = w_mse
        self.w_l1 = w_l1
        self.mse = nn.MSELoss(reduction='mean')
        self.l1 = nn.SmoothL1Loss(reduction='mean', beta=beta_l1)
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss_mse = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        loss_l1 = self.l1(y_pred, y_true)

        loss = loss_mse * self.w_mse + loss_l1 * self.w_l1

        return loss


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss
