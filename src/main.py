# This is the main section of the code, it runs the scripts to train and evaluate the model,
# with each configuration you determine in the experiment_config folder.

from experiment_config.hyper_parameters import (
    MODELS, FOLDS, OPTIMIZERS, LEARNING_RATES, OVERPASS_EXCEPTIONS,
    ONLY_TEST
)
from scripts import train_model, evaluate_model
from utils import error_handler

# The sequence in which the loops appear will determine the sequence in which the tests are done.
# You can swap the loops to change this order.

for model in MODELS:
    for fold in range(1, FOLDS + 1): # starting in 1
        for optimizer in OPTIMIZERS:
            for lr in LEARNING_RATES:
                configuration = {
                    "model": model,
                    "optimizer": optimizer,
                    "fold": fold,
                    "lr": lr
                }

                if (OVERPASS_EXCEPTIONS):
                    try:
                        if not ONLY_TEST: train_model(model, optimizer, fold, lr)
                        evaluate_model(configuration)
                    except Exception as e:
                        error_handler(e)
                else:
                    if not ONLY_TEST: train_model(model, optimizer, fold, lr)
                    evaluate_model(configuration)
