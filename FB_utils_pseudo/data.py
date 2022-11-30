import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


def prepare_loaders(_cfg, folds, fold, curclass=None):
    """
    Prepare and build train and eval data loaders
    """

    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    valid_labels = valid_folds[_cfg.target_cols].values

    train_dataset = ValidDataset(_cfg, train_folds)
    valid_dataset = ValidDataset(_cfg, valid_folds)
    train_bs = _cfg.train_bs

    train_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              shuffle=True,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=_cfg.valid_bs,
                              shuffle=False,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=False)

    print(f'Size of train dataset: {len(train_dataset)}')
    print(f'Size of eval dataset: {len(valid_dataset)}')

    return train_loader, valid_loader, valid_labels, valid_folds


def prepare_pseudo_loader(_cfg, folds):
    """
    Prepare and build train and eval data loaders
    """

    train_dataset = TrainDataset(_cfg, folds)
    train_bs = _cfg.train_bs

    train_loader = DataLoader(train_dataset,
                              batch_size=train_bs,
                              shuffle=True,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)

    print(f'Size of train dataset: {len(train_dataset)}')

    return train_loader


def prepare_only_valid(_cfg, folds):
    valid_dataset = ValidDataset(_cfg, folds)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=_cfg.valid_bs,
                              shuffle=False,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=False)

    print(f'Size of valid dataset: {len(valid_loader)}')
    return valid_loader


def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values
        self.pseudo = df['weight'].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        pseudo = torch.tensor(self.pseudo[item], dtype=torch.float)
        return inputs, label, pseudo


class ValidDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)

        return inputs, label


def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:, :mask_len]
    return inputs


def precollate(inputs, labels, nears, loader):
    nears = nears[0].numpy()
    inp_, lab_ = [], []
    for val in nears:
        val1, val2 = loader.dataset.__getitem__(val, near=False)
        inp_.append(val1)
        lab_.append(val2)

    for k in inputs:
        inputs[k] = torch.cat([inputs[k], torch.stack([x[k] for x in inp_])])

    labels = torch.cat([labels, torch.stack(lab_)])
    return inputs, labels
