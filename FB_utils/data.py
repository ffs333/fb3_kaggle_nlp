import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


def prepare_loaders(_cfg, folds, fold, curclass=None):
    """
    Prepare and build train and eval data loaders
    """

    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)

    if _cfg.num_classes == 1:
        valid_labels = valid_folds[curclass].values.reshape(-1, 1)
    else:
        valid_labels = valid_folds[_cfg.target_cols].values

    if _cfg.normlen:
        train_dataset = TrainDatasetBalancedLen(_cfg, train_folds)
        valid_dataset = TrainDataset(_cfg, valid_folds)
        train_bs = 1
    else:
        if _cfg.num_classes == 1:
            train_dataset = TrainDatasetSingle(_cfg, train_folds, curclass)
            valid_dataset = TrainDatasetSingle(_cfg, valid_folds, curclass)
        else:
            train_dataset = TrainDataset(_cfg, train_folds)
            valid_dataset = TrainDataset(_cfg, valid_folds)
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


class TrainDatasetSingle(Dataset):
    def __init__(self, cfg, df, cur_class):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cur_class].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input(self.cfg, self.texts[item])
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


class TrainDataset(Dataset):
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


class TrainDatasetBalancedLen(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.df = df.copy()
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item, near=True):
        sample = self.df.loc[item]
        inputs = prepare_input(self.cfg, sample['full_text'])
        # label = torch.tensor(self.labels[item], dtype=torch.float)
        label = torch.tensor(sample[self.cfg.target_cols].values.astype('float'), dtype=torch.float)

        if near:
            nearest = self.df.iloc[(self.df['length'] - sample['length']).abs().argsort()[:50]].index.values
            return inputs, label, np.random.choice(nearest, self.cfg.train_bs - 1)
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
