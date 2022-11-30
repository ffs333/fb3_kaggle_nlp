import torch


class CFG:
    # MAIN
    wandb = True
    competition = 'FB3'
    wb_group = 'single_class'
    exp_name = 'debertav3large_V3_signle_class_'
    base_path = '/home/artem/kf/proj/kaggle/fb_eng/'

    seed = 333
    train = True
    debug = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DATA
    target_cols = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

    normlen = False
    num_workers = 8
    train_bs = 3
    valid_bs = 3
    max_len = 512

    n_fold = 6
    trn_fold = [2, 3, 5]  # [0, 1, 2, 3, 4]

    # MODEL
    model = "microsoft/deberta-v3-large"  # microsoft/deberta-v2-xxlarge
    gradient_checkpointing = True
    num_classes = 6

    # TRAIN
    apex = True
    use_restart = True

    # Loss
    loss = 'l1'  # ['l1', 'double', 'rmse']
    w_mse = 0.25
    w_l1 = 0.75
    beta_L1 = 0.125

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

    # STEP 2
    step2 = True

    # Scheduler step 2

    scheduler_step2 = 'cosine_restart'
    num_cycles_step2 = 2.5
    # Loop step 2

    epochs_step2 = 6
    rest_thr_step2 = 0.002  # 0.002
    iter4eval_step2 = 163  # 163

    # LR 2
    lr_step2 = 4e-6  # 2.8e-6


os.makedirs(CFG.base_path + 'results/', exist_ok=True)
os.makedirs(CFG.base_path + 'results/' + CFG.exp_name, exist_ok=True)
os.makedirs(CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints', exist_ok=True)
CFG.save_path = CFG.base_path + 'results/' + CFG.exp_name + '/checkpoints/'
with open(CFG.base_path + 'results/' + CFG.exp_name + '/CFG.txt', 'w') as f:
    for key, value in CFG.__dict__.items():
        f.write('%s:%s\n' % (key, value))