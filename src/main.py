# This is the main section of the code, it runs the scripts to train and evaluate the model,
# with each configuration you determine in the experiment_config folder.

import os
from mmcv.runner.hooks import Hook

from experiment_config.early_stop import checkBest
from scripts import train_model, evaluate_model
from utils import error_handler
from experiment_config.config_cfg import config_cfg
from experiment_config.hyper_parameters import (
    MODELS, FOLDS, OPTIMIZERS, LEARNING_RATES, EPOCHS,
    OVERPASS_EXCEPTIONS, SKIP_TRAIN, EARLY_STOP
)

if (EARLY_STOP): Hook.after_val_epoch = checkBest

data_root = os.path.join(os.getcwd(), 'dataset')

# The sequence in which the loops appear will determine the sequence in which the tests are done.
# You can swap the loops to change this order.

for model in MODELS.keys():
    for fold in range(1, FOLDS + 1): # starting in 1
        for optimizer in OPTIMIZERS:
            for lr in LEARNING_RATES:
                # The cfg contains all the model configuraton
                cfg = config_cfg(data_root, EPOCHS, model, lr, fold, optimizer)

                configuration = {
                    "model": model,
                    "optimizer": optimizer,
                    "fold": fold,
                    "lr": lr,
                    "cfg": cfg
                }

                if (OVERPASS_EXCEPTIONS):
                    try:
                        if not SKIP_TRAIN: train_model(configuration)
                        evaluate_model(configuration)
                    except Exception as e:
                        error_handler(e)
                else:
                    if not SKIP_TRAIN: train_model(configuration)
                    evaluate_model(configuration)
