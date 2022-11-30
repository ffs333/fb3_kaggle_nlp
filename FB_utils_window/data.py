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

    train_dataset = TrainDataset(_cfg, train_folds)
    valid_dataset = ValidDataset(_cfg, valid_folds)

    train_loader = DataLoader(train_dataset,
                              batch_size=_cfg.train_bs,
                              shuffle=True,
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=_cfg.valid_bs,
                              shuffle=False,
                              collate_fn=CollatorValidMulti(_cfg.tokenizer),
                              num_workers=_cfg.num_workers, pin_memory=True, drop_last=False)

    print(f'Size of train dataset: {len(train_dataset)}')
    print(f'Size of eval dataset: {len(valid_dataset)}')

    return train_loader, valid_loader, valid_labels, valid_folds


def prepare_input_more(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text,
        return_tensors=None,
        add_special_tokens=True,
        # max_length=cfg.max_len,
        # pad_to_max_length=True,
        # truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input_more(self.cfg, self.texts[item])

        if len(inputs['input_ids']) > 510:
            start_ind = np.random.randint(0, len(inputs['input_ids']) - 510)

            for k, v in inputs.items():
                inputs[k] = v[start_ind:start_ind+510]
                if k == 'input_ids':
                    inputs[k] = torch.cat([torch.tensor([1]), inputs[k], torch.tensor([2])])
                elif k == 'token_type_ids':
                    inputs[k] = torch.cat([inputs[k], torch.tensor([0, 0])])
                elif k == 'attention_mask':
                    inputs[k] = torch.cat([inputs[k], torch.tensor([1, 1])])
                else:
                    raise ValueError('WRONG KEY FOR INPUTS')
        else:
            inputs = prepare_input(self.cfg, self.texts[item])

        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label


class CollatorValidMulti:
    def __init__(self, tokeniz):
        self.tokenizer = tokeniz

    def __call__(self, batch):

        labels = [] #torch.stack([v[1] for v in batch])
        length_ = torch.tensor([v[2] for v in batch])
        
        full_x = []
        for text_ in batch:
            for part in text_[0]:
                labels.append(text_[1])
                full_x.append(part)

        text_batch = self.tokenizer.pad(full_x,
                                        max_length=512,
                                        pad_to_multiple_of=None,
                                        return_tensors='pt',
                                        )
        labels = torch.stack(labels)

        return text_batch, labels, length_


class ValidDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.texts = df['full_text'].values
        self.labels = df[cfg.target_cols].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        inputs = prepare_input_more(self.cfg, self.texts[item])

        len_data = len(inputs['input_ids'])
        if len_data > self.cfg.window_size + 20:
            ind_ar = []
            full_data = []
            for ij in range(0, len_data, self.cfg.window_size - self.cfg.window_step):
                ind_ar.append((max(min(ij, len_data - self.cfg.window_size), 0),
                               min(len_data, ij + self.cfg.window_size)))
            ind_ar = np.unique(np.array(ind_ar), axis=0)

            for ar in ind_ar:
                cur_inp = dict()
                for k, v in inputs.items():
                    cur_inp[k] = v[ar[0]:ar[1]]
                    if k == 'input_ids':
                        cur_inp[k] = torch.cat([torch.tensor([1]), cur_inp[k], torch.tensor([2])])
                    elif k == 'token_type_ids':
                        cur_inp[k] = torch.cat([cur_inp[k], torch.tensor([0, 0])])
                    elif k == 'attention_mask':
                        cur_inp[k] = torch.cat([cur_inp[k], torch.tensor([1, 1])])
                    else:
                        raise ValueError('WRONG KEY FOR INPUTS')
                full_data.append(cur_inp)

        else:
            inputs = prepare_input(self.cfg, self.texts[item])
            full_data = [inputs]

        label = torch.tensor(self.labels[item], dtype=torch.float)
        return full_data, label, len(full_data)


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
