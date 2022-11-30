import gc

from tqdm.auto import tqdm
import torch
import wandb
import numpy as np
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from .data import collate, precollate
from .utils import AverageMeter, get_score, FGM


def get_optimizer(model, cfg, step2=False):
    if step2:
        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=cfg.lr_step2,
                                                    decoder_lr=cfg.lr_step2,
                                                    weight_decay=cfg.weight_decay_step2)
        if cfg.optimizer == 'Adam':
            optimizer = optim.Adam(optimizer_parameters, lr=cfg.lr_step2, eps=cfg.eps_step2, betas=cfg.betas_step2)
        elif cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(optimizer_parameters, lr=cfg.lr_step2, eps=cfg.eps_step2, betas=cfg.betas_step2)
        else:
            raise ValueError('Error in "get_optimizer" function:',
                             f'Wrong optimizer name. Choose one from ["Adam", "AdamW"] ')

    else:

        optimizer_parameters = get_optimizer_params(model,
                                                    encoder_lr=cfg.encoder_lr,
                                                    decoder_lr=cfg.decoder_lr,
                                                    weight_decay=cfg.weight_decay)
        if cfg.optimizer == 'Adam':
            optimizer = optim.Adam(optimizer_parameters, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas)
        elif cfg.optimizer == 'AdamW':
            optimizer = optim.AdamW(optimizer_parameters, lr=cfg.encoder_lr, eps=cfg.eps, betas=cfg.betas)
        else:
            raise ValueError('Error in "get_optimizer" function:',
                             f'Wrong optimizer name. Choose one from ["Adam", "AdamW"] ')

    return optimizer


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': encoder_lr, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if "model" not in n],
         'lr': decoder_lr, 'weight_decay': 0.0}
    ]
    return optimizer_parameters


def get_scheduler(cfg, scheduler_name, optimizer, num_train_steps, cycles):
    if scheduler_name == 'linear':
        scheduler = get_linear_schedule_with_warmup(
                                                    optimizer, num_warmup_steps=cfg.num_warmup_steps,
                                                    num_training_steps=num_train_steps)
    elif scheduler_name == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
                                                    optimizer, num_warmup_steps=cfg.num_warmup_steps,
                                                    num_training_steps=num_train_steps, num_cycles=cycles)

    elif scheduler_name == 'cosine_restart':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                   T_0=int(num_train_steps // cycles), T_mult=1)

    else:
        raise ValueError('Error in "get_scheduler" function:',
                         f'Wrong scheduler name. Choose one from ["linear", "cosine", "cosine_restart" ]')

    return scheduler


def train_fn(cfg, fold, train_loader, valid_loader, model, criterion, optimizer, scheduler, device, epoch,
             valid_labels, LOGGER, best_score, valid_points,
             _global_step, _it4eval, save_path, curclass=None, step2=False):

    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.apex)
    losses = AverageMeter()

    if curclass:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Ep.{epoch + 1} [{curclass}] Train')
    else:
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Ep.{epoch + 1} Train ')

    if cfg.use_fgm:
        fgm = FGM(model)

    for step, batch in pbar:
        _global_step += 1

        if cfg.pseudo_training:
            inputs, labels, pseudo = batch
            pseudo = pseudo.to(device)
        else:
            inputs, labels = batch

        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        if cfg.num_classes == 1:
            labels = labels.unsqueeze(1)

        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=cfg.apex):
            y_preds = model(inputs)
            if cfg.pseudo_training:
                loss = criterion(y_preds, labels, pseudo)
            else:
                loss = criterion(y_preds, labels)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        if cfg.use_fgm:
            fgm.attack()
            with torch.cuda.amp.autocast(enabled=cfg.apex):
                y_preds = model(inputs)
                if cfg.pseudo_training:
                    loss_adv = criterion(y_preds, labels, pseudo)
                else:
                    loss_adv = criterion(y_preds, labels)
            if cfg.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / cfg.gradient_accumulation_steps
            scaler.scale(loss_adv).backward()
            fgm.restore()

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
            wandb.log({f"[fold{fold}, {curclass}] loss": losses.val,
                       f"[fold{fold}, {curclass}] lr": current_lr})

        torch.cuda.empty_cache()
        gc.collect()

        if (step != 0 and _global_step % _it4eval == 0 and len(train_loader) - step > 17) or \
                (_global_step in valid_points):

            avg_val_loss, predictions = valid_fn(cfg=cfg, valid_loader=valid_loader, model=model, epoch=epoch,
                                                 criterion=criterion, device=device, curclass=curclass)

            score, scores = get_score(valid_labels, predictions)
            LOGGER.info(f'Epoch {epoch + 1} , step {_global_step} - Score: {score:.4f} ')

            if best_score >= score:
                LOGGER.info(f'Best Score Updated {best_score:0.4f} -->> {score:0.4f} | Model Saved')
                best_score = score

                torch.save({'model': model.state_dict(),
                            'predictions': predictions},
                           save_path)
            else:
                LOGGER.info(f'Score NOT updated. Current best: {best_score:0.4f}')
                rest_thr_ = cfg.rest_thr_step2 if step2 else cfg.rest_thr
                if cfg.use_restart and score - best_score > rest_thr_:
                    loaded_check = torch.load(save_path,
                                              map_location=torch.device('cpu'))
                    model.to(torch.device('cpu'))
                    model.load_state_dict(loaded_check['model'])
                    model.to(device)
                    LOGGER.info('Loaded previous best model')

            if cfg.wandb:
                wandb.log({f"[fold{fold}, {curclass}] score": score,
                           f"[fold{fold}, {curclass}] avg_train_loss": losses.avg,
                           f"[fold{fold}, {curclass}] avg_val_loss": avg_val_loss})

            model.train()

    return losses.avg, best_score


@torch.no_grad()
def valid_fn(cfg, valid_loader, model, epoch, criterion, device, curclass=None):
    losses = AverageMeter()
    model.eval()
    preds = []

    if curclass:
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Ep.{epoch + 1} [{curclass}] Valid')
    else:
        pbar = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Ep.{epoch + 1} Valid ')

    for step, batch in pbar:
        
        if cfg.pseudo_training:
            inputs, labels, pseudo = batch
            pseudo = pseudo.to(device)
        else:
            inputs, labels = batch
            
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        if cfg.num_classes == 1:
            labels = labels.unsqueeze(1)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            y_preds = model(inputs)
            if cfg.pseudo_training:
                loss = criterion(y_preds, labels, pseudo)
            else:
                loss = criterion(y_preds, labels)
        if cfg.gradient_accumulation_steps > 1:
            loss = loss / cfg.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.to('cpu').numpy())

        mem = torch.cuda.memory_reserved(f'cuda') / 1E9 if torch.cuda.is_available() else 0

        pbar.set_postfix(_loss=f'{losses.avg:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')

        torch.cuda.empty_cache()
        gc.collect()

    predictions = np.concatenate(preds)
    return losses.avg, predictions
