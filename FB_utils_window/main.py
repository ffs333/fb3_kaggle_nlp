# ====================================================
# Imports
# ====================================================
import os
import warnings
warnings.filterwarnings("ignore")

import wandb
import torch
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from FB_utils.pipeline import train_loop
from models import get_tokenizer
from utils import class2dict, get_logger, define_max_len, get_result

# %env TOKENIZERS_PARALLELISM=true
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class CFG:
    ####################
    # MAIN
    ####################
    wandb = False
    wandb_project = 'FeedBack_kaggle'
    competition = 'FB3'
    wb_group = 'single_class'
    exp_name = 'debertav3large_V3_single_class_'
    base_path = '/Users/kolyan/downloads/FB_KAGL/code/' #'/home/artem/kf/proj/kaggle/fb_eng/'

    seed = 333
    train = True
    debug = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ####################
    # DATA
    ####################
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    normlen = False
    num_workers = 0 #8
    train_bs = 3
    valid_bs = 3
    max_len = 512

    n_fold = 6
    trn_fold = [2, 3, 5]  # [0, 1, 2, 3, 4]

    ####################
    # MODEL
    ####################
    model = "microsoft/deberta-v3-large"  # microsoft/deberta-v2-xxlarge
    gradient_checkpointing = True
    num_classes = 6

    ####################
    # TRAIN
    ####################
    apex = True
    use_restart = True

    ####################
    # LOSS
    ####################
    loss = 'l1'  # ['l1', 'double', 'rmse']
    w_mse = 0.25
    w_l1 = 0.75
    beta_L1 = 0.125

    ####################
    # STEP 1
    ####################

    # Scheduler step 1
    scheduler = 'cosine'  # ['linear', 'cosine']
    num_cycles = 0.5  # 3.5
    num_warmup_steps = 100

    # Loop step 1
    epochs = 3  # 6
    rest_thr = 0.004  # 0.012
    iter4eval = 339

    # LR, optimizer step 1
    encoder_lr = 1.3e-5  # 1.5e-5 # 2e-5
    decoder_lr = 1.3e-5  # 1.5e-5 # 2e-5
    min_lr = 0.01e-6  # 1e-6
    eps = 1e-6
    betas = (0.9, 0.999)
    weight_decay = 0.0001
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    optimizer = 'AdamW'

    ####################
    # STEP 2
    ####################
    step2 = True

    # Scheduler step 2
    scheduler_step2 = 'cosine_restart'
    num_cycles_step2 = 2.5

    # Loop step 2
    epochs_step2 = 6
    rest_thr_step2 = 0.002  # 0.002
    iter4eval_step2 = 163  # 163

    # LR step 2
    lr_step2 = 4e-6  # 2.8e-6


os.makedirs(CFG.base_path + 'results/', exist_ok=True)
os.makedirs(CFG.base_path + 'results/' + CFG.exp_name, exist_ok=True)
os.makedirs(CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints', exist_ok=True)
CFG.save_path = CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints/'
with open(CFG.base_path + 'results/' + CFG.exp_name + '/CFG.txt', 'w') as f:
    for key, value in CFG.__dict__.items():
        f.write('%s:%s\n' % (key, value))

if CFG.wandb:
    wandb.init(project=CFG.wandb_project,
               name=CFG.exp_name,
               config=class2dict(CFG),
               group=CFG.wb_group,
               job_type="train",
               dir=CFG.base_path)

LOGGER = get_logger(CFG.base_path + 'results/' + CFG.exp_name + '/train')


train = pd.read_csv(f'{CFG.base_path}/feedback-prize-english-language-learning/train.csv')
test = pd.read_csv(f'{CFG.base_path}/feedback-prize-english-language-learning/test.csv')
submission = pd.read_csv(f'{CFG.base_path}/feedback-prize-english-language-learning/sample_submission.csv')

print(f"train.shape: {train.shape}")
print(f"test.shape: {test.shape}")
print(f"submission.shape: {submission.shape}")


# ====================================================
# tokenizer & define max len
# ====================================================
CFG.tokenizer = get_tokenizer(CFG)
max_len, lengths = define_max_len(train, CFG.tokenizer)
CFG.max_len = max_len
train['length'] = lengths


# ====================================================
# CV split
# ====================================================
Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=42)
for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)
#display(train.groupby('fold').size())


if CFG.debug:
    #display(train.groupby('fold').size())
    train = train.sample(n=150, random_state=0).reset_index(drop=True)
    #display(train.groupby('fold').size())

if __name__ == '__main__':

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                if CFG.num_classes == 1:
                    for ind_cl, curclass in enumerate(CFG.target_cols):
                        _oof_df = train_loop(CFG=CFG, folds=train, fold=fold, LOGGER=LOGGER, curclass=curclass)
                        if ind_cl == 0:
                            fold_df = _oof_df.copy()
                        else:
                            fold_df = pd.merge(fold_df, _oof_df,
                                               on=['text_id', 'full_text', 'length', 'fold'] + CFG.target_cols)
                    oof_df = pd.concat([oof_df, fold_df])
                    LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(fold_df, CFG, LOGGER)
                else:
                    _oof_df = train_loop(CFG=CFG, folds=train, fold=fold, LOGGER=LOGGER, curclass=None)
                    oof_df = pd.concat([oof_df, _oof_df])
                    LOGGER.info(f"========== fold: {fold} result ==========")
                    get_result(_oof_df, CFG, LOGGER)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df, CFG, LOGGER)
        oof_df.to_pickle(CFG.save_path + 'oof_df.pkl')

    if CFG.wandb:
        wandb.finish()
