import gc
import time

import torch
import wandb
import numpy as np

from FB_utils.core import get_optimizer, get_scheduler, train_fn, valid_fn
from .utils import get_score, set_seed
from .data import prepare_loaders
from .models import CustomModel
from .losses import get_loss_func
from .sched_find import gpfs_


def fine_train_loop(CFG, folds, fine_dct, LOGGER, curclass=None):

    set_seed(CFG.seed)

    _model_name, fold, best_score, _config, ckpt_path = fine_dct.values()

    LOGGER.info(f"========== Fold: {fold} training ==========")

    train_loader, valid_loader, valid_labels, valid_folds = prepare_loaders(CFG, folds, fold, curclass)

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================

    model = CustomModel(CFG, config_path=_config, pretrained=False)
    torch.save(model.config, CFG.save_path + 'config.pth')
    loaded_check = torch.load(ckpt_path,
                              map_location=torch.device('cpu'))
    model.load_state_dict(loaded_check['model'])
    model.to(CFG.device)

    save_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_fold{fold}_FINETUNED_best.pth"
    torch.save({'model': model.state_dict(),
                'predictions': loaded_check['predictions']}, save_path)

    optimizer = get_optimizer(model, CFG, step2=False)

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    _global_step = 0
    valid_points = gpfs_(CFG, len(train_loader))
    if len(CFG.valid_pnts) > 0:
        valid_points = np.append(valid_points, CFG.valid_pnts)
    print(f'Validation points: {valid_points}')

    for epoch in range(CFG.epochs):

        start_time = time.time()
        print(f'Epoch {epoch + 1}/{CFG.epochs} | Fold {fold} | Class {curclass}')

        # train
        avg_loss, best_score = train_fn(cfg=CFG, fold=fold, train_loader=train_loader, valid_loader=valid_loader,
                                        model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                        device=CFG.device, epoch=epoch, valid_labels=valid_labels, LOGGER=LOGGER,
                                        best_score=best_score, valid_points=valid_points, _global_step=_global_step,
                                        _it4eval=CFG.iter4eval, save_path=save_path,
                                        curclass=curclass, step2=False)

        _global_step += len(train_loader)

        # eval
        avg_val_loss, predictions = valid_fn(cfg=CFG, valid_loader=valid_loader, model=model, epoch=epoch,
                                             criterion=criterion, device=CFG.device, curclass=curclass)

        # scoring
        score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  '
            f'avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}')
        if CFG.wandb:
            wandb.log({f"[fold{fold}, {curclass}] score": score,
                       f"[fold{fold}, {curclass}] avg_train_loss": avg_loss,
                       f"[fold{fold}, {curclass}] avg_val_loss": avg_val_loss})

        if best_score >= score:
            LOGGER.info(f'Best Score Updated {best_score:0.4f} -->> {score:0.4f} | Model Saved')
            best_score = score
            LOGGER.info(f'Epoch {epoch + 1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(),
                        'predictions': predictions},
                       save_path)
        else:
            LOGGER.info(f'Score NOT updated. Current best: {best_score:0.4f}')
            if CFG.use_restart and score - best_score > CFG.rest_thr:
                loaded_check = torch.load(save_path,
                                          map_location=torch.device('cpu'))
                model.to(torch.device('cpu'))
                model.load_state_dict(loaded_check['model'])
                model.to(CFG.device)
                LOGGER.info('Loaded previous best model')

    loadeed_check = torch.load(save_path,
                               map_location=torch.device('cpu'))
    predictions = loadeed_check['predictions']

    final_s_path = CFG.save_path+f"{CFG.model.replace('/', '-')}_fold{fold}_final_FINETUNED_best_{best_score:0.5f}.pth"

    torch.save({'model': loadeed_check['model'],
                'predictions': predictions},
               final_s_path)

    LOGGER.info(f'FOLD {fold} TRAINING FINISHED. BEST SCORE: {best_score:0.4f}',
                f'SAVED HERE: {final_s_path}')

    if CFG.num_classes == 1:
        valid_folds[f"pred_{curclass}"] = predictions
    else:
        valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions

    torch.cuda.empty_cache()
    gc.collect()

    return valid_folds
