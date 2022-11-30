import gc

from tqdm.auto import tqdm
import torch
import wandb
import numpy as np

from .data import collate, precollate
from .utils import AverageMeter, get_score


def train_fn_pseudo(cfg, train_loader, model, criterion, optimizer, scheduler, device, epoch,
                    loop, _global_step):

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Ep.{epoch + 1} Train ')

    for step, batch in pbar:
        _global_step += 1
        inputs, labels, pseudo = batch

        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        pseudo = pseudo.to(device)

        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels, pseudo)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']

        pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                         lr=f'{current_lr:0.8f}',
                         gpu_mem=f'{mem:0.2f} GB',
                         global_step=f'{_global_step}')

        if cfg.wandb:
            wandb.log({f"[Loop{loop}] loss": losses.val,
                       f"[Loop{loop}] lr": current_lr})

        torch.cuda.empty_cache()
        gc.collect()

    return losses.avg


from .data import prepare_only_valid
from .utils import get_score, set_seed
from .models import CustomModel
from .utils import get_score, set_seed

@torch.no_grad()
def valid_fn_pseudo(CFG, folds, loop, LOGGER, checkpoint):

    set_seed(CFG.seed)

    LOGGER.info(f"========== Pseudo Loop {loop} Validation ==========")

    valid_loader = prepare_only_valid(CFG, folds)

    model = CustomModel(CFG, config_path=None, pretrained=True)
    loadeed_check = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.to(torch.device('cpu'))
    model.load_state_dict(loadeed_check['model'])
    model.to(CFG.device)
    model.eval()

    preds = []

    pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Valid ')

    for step, (inputs, labels) in pbar:
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(CFG.device)

        with torch.no_grad():
            y_preds = model(inputs)

        preds.append(y_preds.to('cpu').numpy())
        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(gpu_mem=f'{mem:0.2f} GB')

        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.concatenate(preds)
    score, scores = get_score(folds[CFG.target_cols].values, predictions)
    LOGGER.info(f'Validation - Score: {score:.4f}  Scores: {scores}')
    return predictions , score, scores
