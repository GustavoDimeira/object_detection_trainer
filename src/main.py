# This is the main section of the code, it runs the scripts to train and evaluate the model,
# with each configuration you determine in the experiment_config folder.

import os, wget
from mmcv.runner.hooks import Hook

from scripts.train_model import train_model
# from scripts.evaluate_model import evaluate_model
from utils.error_handler import error_handler
from experiment_config.early_stop import checkBest
from experiment_config.config_cfg import config_cfg
from experiment_config.hyper_parameters import (
    MODELS, FOLDS, OPTIMIZERS, LEARNING_RATES, EPOCHS,
    OVERPASS_EXCEPTIONS, SKIP_TRAIN, EARLY_STOP, SAVE_IMAGES
)

if (EARLY_STOP): Hook.after_val_epoch = checkBest

MODELS_KEYS = [k for k in MODELS]
data_root = os.path.join(os.getcwd(), 'results')

if not os.path.exists("checkpoints"):
    print('\nCriando diretório checkpoints...')
    os.makedirs('checkpoints')

for model in MODELS:
    if not os.path.exists(MODELS[model]['checkpoint']):
        print(f"\nBaixando checkpoint do modelo {model}.")
        wget.download(MODELS[model]["model_download"], out="checkpoints")

# The sequence in which the loops appear will determine the sequence in which the tests are done.
# You can swap the loops to change this order.

combinations = []
for fold in range(1, FOLDS + 1): # starting in 1
    for lr in LEARNING_RATES:
        for optimizer in OPTIMIZERS:
            for model in MODELS_KEYS:
                combinations.append([fold, lr, optimizer, model])

for i in range(len(combinations)):
    fold, lr, optimizer, model = combinations[i]
    # The cfg contains all the model configuraton
    cfg = config_cfg(data_root, EPOCHS, model, lr, fold, optimizer)

    configuration = {
        "model": model,
        "optimizer": optimizer,
        "fold": fold,
        "lr": lr,
        "cfg": cfg
    }

    print("Currently configuration.")
    print(model, fold, optimizer, lr, '\n')

    if (OVERPASS_EXCEPTIONS):
        try:
            if not SKIP_TRAIN: train_model(cfg, model)
            # evaluate_model(cfg=cfg, models_path=pth, show_imgs=False,
            #        save_imgs=SAVE_IMAGES, fold=fold, model_name=model)
        except Exception as e:
            error_handler(e)
    else:
        if not SKIP_TRAIN: train_model(cfg, model)
        # evaluate_model(cfg=cfg, models_path=pth, show_imgs=False,
        #        save_imgs=SAVE_IMAGES, fold=fold, model_name=model)
