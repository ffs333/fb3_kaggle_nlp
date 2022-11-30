from torch import nn
import torch
import numpy as np
from scipy.signal import find_peaks

from FB_utils.core import get_scheduler


def gpfs_(cfg, len_load):
    all_steps_s1 = cfg.epochs * len_load
    if cfg.step2:
        all_steps_s2 = cfg.epochs_step2 * len_load

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)

        def forward(self, x):
            return self.fc(x)

    mdl = Model()
    opt = torch.optim.AdamW(mdl.parameters(), lr=cfg.encoder_lr)

    # len_load = 1086
    sch_s1 = get_scheduler(cfg, cfg.scheduler, opt, len_load*cfg.epochs, cfg.num_cycles)

    full_lr = []
    for i in range(all_steps_s1):
        current_lr = sch_s1.get_lr()[0]
        opt.step()
        sch_s1.step()
        full_lr.append(current_lr)

    full_lr = np.array(full_lr) * -1
    peaks1, _ = find_peaks(full_lr, height=-1)
    peaks1 = np.append(peaks1, len(full_lr) - 1)

    full_peaks1 = peaks1
    full_peaks1 = full_peaks1[(full_peaks1 > 100) & (full_peaks1 < (all_steps_s1 - 100))]

    if cfg.step2:
        mdl = Model()
        opt = torch.optim.AdamW(mdl.parameters(), lr=cfg.lr_step2)

        sch_s2 = get_scheduler(cfg, cfg.scheduler_step2, opt, len_load * cfg.epochs_step2, cfg.num_cycles_step2)

        full_lr = []
        for i in range(all_steps_s2):  #
            current_lr = sch_s2.get_lr()[0]
            opt.step()
            sch_s2.step()
            full_lr.append(current_lr)

        full_lr = np.array(full_lr) * -1
        peaks2, _ = find_peaks(full_lr, height=-1)
        peaks2 = np.append(peaks2, len(full_lr) - 1)

        full_peaks2 = peaks2
        full_peaks2 = full_peaks2[(full_peaks2 > 100) & (full_peaks2 < (all_steps_s2 - 100))]
        full_peaks2 = full_peaks2 + all_steps_s1

        return np.concatenate([full_peaks1, full_peaks2])
    return full_peaks1
