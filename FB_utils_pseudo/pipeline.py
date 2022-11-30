import gc
import time

import torch
import wandb
import numpy as np

from .core import get_optimizer, get_scheduler, train_fn, valid_fn
from .core_pseudo import train_fn_pseudo
from .utils import get_score, set_seed
from .data import prepare_loaders, prepare_pseudo_loader
from .models import CustomModel
from .losses import get_loss_func
from .sched_find import gpfs_


def train_loop(CFG, folds, fold, LOGGER, curclass=None):

    set_seed(CFG.seed)

    if curclass:
        LOGGER.info(f"========== Fold: {fold} for class {curclass} training ==========")
    else:
        LOGGER.info(f"========== Fold: {fold} training ==========")

    train_loader, valid_loader, valid_labels, valid_folds = prepare_loaders(CFG, folds, fold, curclass)

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================

    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG.save_path + 'config.pth')
    model.to(CFG.device)

    optimizer = get_optimizer(model, CFG, step2=False)

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    best_score = np.inf
    _global_step = 0
    valid_points = gpfs_(CFG, len(train_loader))
    if len(CFG.valid_pnts) > 0:
        valid_points = np.append(valid_points, CFG.valid_pnts)
    print(f'Validation points: {valid_points}')

    if curclass:
        save_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_fold{fold}_{curclass}_best.pth"
    else:
        save_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"

    for epoch in range(CFG.epochs):

        start_time = time.time()
        print(f'Epoch {epoch + 1}/{CFG.epochs} | Fold {fold} | Class {curclass}')
        #########
        if fold == 0 and False:
            
            save_path = CFG.save_path + 'microsoft-deberta-v3-large_fold0_best_SAVEBEST_0.4432.pth'
            print(f'loaded this {save_path}')
            loaded_check = torch.load(save_path,
                                      map_location=torch.device('cpu'))
            model.to(torch.device('cpu'))
            model.load_state_dict(loaded_check['model'])
            model.to(CFG.device)

            best_score = 0.4432
            
            print(f'skipped step1 for fold 0')
            _global_step += len(train_loader) * CFG.epochs

            # eval
            avg_val_loss, predictions = valid_fn(cfg=CFG, valid_loader=valid_loader, model=model, epoch=epoch,
                                                 criterion=criterion, device=CFG.device, curclass=curclass)

            # scoring
            score, scores = get_score(valid_labels, predictions)

            LOGGER.info(
                
                f'avg_val_loss: {avg_val_loss:.4f}  ')
            LOGGER.info(f'Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}')
            if CFG.wandb:
                wandb.log({f"[fold{fold}, {curclass}] score": score,
                          f"[fold{fold}, {curclass}] avg_val_loss": avg_val_loss})
            break
        ##########

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

    # ====================================================
    # Step 2
    # ====================================================

    if CFG.step2:
        loaded_check = torch.load(save_path, map_location=torch.device('cpu'))
        model.to(torch.device('cpu'))
        model.load_state_dict(loaded_check['model'])
        model.to(CFG.device)
        if curclass:
            save_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_fold{fold}_{curclass}_best_step2.pth"
        else:
            save_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_fold{fold}_best_step2.pth"
        print(f'step2 save_path: {save_path}')
        LOGGER.info('STARTING STEP 2')
        torch.save({'model': model.state_dict(),
                    'predictions': predictions},
                   save_path)

        optimizer = get_optimizer(model, CFG, step2=True)

        num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                              CFG.gradient_accumulation_steps * CFG.epochs_step2)
        scheduler = get_scheduler(CFG, CFG.scheduler_step2, optimizer, num_train_steps, CFG.num_cycles_step2)

        for epoch in range(CFG.epochs_step2):

            start_time = time.time()

            print(f'Epoch {epoch + 1}/{CFG.epochs_step2} | Fold {fold} | Step 2')

            # train
            avg_loss, best_score = train_fn(cfg=CFG, fold=fold, train_loader=train_loader, valid_loader=valid_loader,
                                            model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                                            device=CFG.device, epoch=epoch, valid_labels=valid_labels, LOGGER=LOGGER,
                                            best_score=best_score, valid_points=valid_points, _global_step=_global_step,
                                            _it4eval=CFG.iter4eval_step2, save_path=save_path,
                                            curclass=curclass, step2=True)

            _global_step += len(train_loader)

            # eval
            avg_val_loss, predictions = valid_fn(cfg=CFG, valid_loader=valid_loader, model=model, epoch=epoch,
                                                 criterion=criterion, device=CFG.device, curclass=curclass)

            # scoring
            score, scores = get_score(valid_labels, predictions)

            elapsed = time.time() - start_time

            LOGGER.info(
                f'STEP 2 | Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  '
                f'avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'STEP 2 | Epoch {epoch + 1} - Score: {score:.4f}  Scores: {scores}')
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
                if CFG.use_restart and score - best_score > CFG.rest_thr_step2:
                    loadeed_check = torch.load(save_path,
                                               map_location=torch.device('cpu'))
                    model.to(torch.device('cpu'))
                    model.load_state_dict(loadeed_check['model'])
                    model.to(CFG.device)
                    LOGGER.info('Loaded previous best model')

                    # END OF TRAINING

    loadeed_check = torch.load(save_path,
                               map_location=torch.device('cpu'))
    predictions = loadeed_check['predictions']

    if curclass:
        final_s_path = CFG.save_path \
                       + f"{CFG.model.replace('/', '-')}_fold{fold}_{curclass}_final_best_{best_score:0.5f}.pth"
    else:
        final_s_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_fold{fold}_final_best_{best_score:0.5f}.pth"

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


def train_loop_pseudo(CFG, folds, loop, LOGGER):
    set_seed(CFG.seed)

    LOGGER.info(f"========== Pseudo Loop {loop} Training ==========")

    train_loader = prepare_pseudo_loader(CFG, folds)

    # ====================================================
    # model & optimizer & scheduler & loss
    # ====================================================

    model = CustomModel(CFG, config_path=None, pretrained=True)
    torch.save(model.config, CFG.save_path + 'config.pth')
    model.to(CFG.device)

    optimizer = get_optimizer(model, CFG, step2=False)

    num_train_steps = int(len(train_loader.dataset) / train_loader.batch_size /
                          CFG.gradient_accumulation_steps * CFG.epochs)
    scheduler = get_scheduler(CFG, CFG.scheduler, optimizer, num_train_steps, CFG.num_cycles)

    criterion = get_loss_func(CFG)

    # ====================================================
    # Training loop
    # ====================================================

    _global_step = 0

    save_path = CFG.save_path + f"{CFG.model.replace('/', '-')}_pseudoloop_{loop}"

    for epoch in range(CFG.epochs):

        start_time = time.time()
        print(f'Epoch {epoch + 1}/{CFG.epochs} | Loop {loop} ')

        # train
        avg_loss = train_fn_pseudo(cfg=CFG, train_loader=train_loader, model=model, criterion=criterion,
                                    optimizer=optimizer, scheduler=scheduler,  device=CFG.device, epoch=epoch,
                                    loop=loop, _global_step=_global_step)

        _global_step += len(train_loader)

        elapsed = time.time() - start_time

        LOGGER.info(
            f'Epoch {epoch + 1} - avg_train_loss: {avg_loss:.4f}  '
            f'time: {elapsed:.0f}s')

        if CFG.wandb:
            wandb.log({f"[Loop{loop}] avg_train_loss": avg_loss})

        cur_save_path = save_path + f'_epoch{epoch}.pth'
        torch.save({'model': model.state_dict()}, cur_save_path)

    LOGGER.info(f'Loop {loop} TRAINING FINISHED.'
                f'SAVED HERE: {cur_save_path}')

    torch.cuda.empty_cache()
    gc.collect()
