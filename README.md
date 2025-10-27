# Automated Object Detection Model Trainer

This repository provides a framework to automate the training and evaluation of different object detection models from the [MMDetection](https://github.com/open-mmlab/mmdetection) library. The main goal is to facilitate hyperparameter tuning and model comparison through a systematic grid search approach.

## Key Features

- **Automated Grid Search:** Systematically tests multiple combinations of models, learning rates, and optimizers to find the best-performing configuration.
- **MMDetection Integration:** Leverages the power and flexibility of the MMDetection framework for building, training, and evaluating state-of-the-art object detection models.
- **Early Stopping:** Automatically stops the training process if the model's performance on the validation set ceases to improve, saving significant time and computational resources.
- **Automated Checkpoint Management:** Downloads pre-trained model weights and intelligently saves only the best-performing checkpoints from each experiment, automatically cleaning up unnecessary files.
- **Detailed Logging:** Each experiment generates a dedicated folder containing the configuration file used, a detailed log of the training process, and the resulting model weights.
- **Modular Configuration:** Easily configure all aspects of the experiments—from hyperparameters to model selection—in a centralized and user-friendly file.
- **Advanced Evaluation:** Includes scripts to evaluate model performance using standard metrics like mAP, True Positives (TP), and False Positives (FP), with support for tilling/stitching to handle high-resolution images.

## Project Structure

```
├── instalations
│   ├── install.R
│   └── install.sh
├── README.md
└── src
    ├── dataset
    │   ├── data_aumentation
    │   │   ├── bbox_segmentation.py
    │   │   ├── noise.py
    │   │   └── tilling.py
    │   └── get_folds.py
    ├── experiment_config
    │   ├── config_cfg.py
    │   ├── early_stop.py
    │   └── hyper_parameters.py
    ├── log.json
    ├── main.py
    ├── scripts
    │   ├── evaluate_model.py
    │   └── train_model.py
    └── utils
        ├── error_handler.py
        ├── get_grafics.R
        ├── models_base_config.py
        ├── optmizers.py
        └── terminal_handler.py
```

## How to Use

### 1. Installation
Run the installation scripts located in the `instalations/` directory to set up the environment and install dependencies like MMDetection, PyTorch, and MMCV.

```bash
bash instalations/install.sh
```


### 2. Dataset Preparation
- Place your dataset images in a designated folder.
- Ensure your annotations are in COCO format (`.json`).
- Structure your dataset and create data folds as expected by the scripts in `src/dataset/`. The code automatically discovers classes from your `_annotations.coco.json` file.

### 3. Configure the Experiment
Open `src/experiment_config/hyper_parameters.py` to define the scope of your experiments:

- **`MODELS`**: Add or remove models from the MMDetection library that you want to test.
- **`OPTIMIZERS`**: Define the list of optimizers (e.g., "SGD", "AdamW").
- **`LEARNING_RATES`**: Specify the learning rates to be tested.
- **`EPOCHS`**: Set the maximum number of training epochs.
- **`EARLY_STOP`**: Set to `True` to enable early stopping.

### 4. Run the Training
Execute the main script to start the automated training process. The script will iterate through all combinations of parameters defined in the previous step.

```bash
python src/main.py
```

The results, including logs and the best model checkpoint for each experiment, will be saved in the `results/` directory (created automatically), organized by fold and hyperparameter combination.

## Evaluation
The `src/scripts/evaluate_model.py` script can be used to assess the performance of a trained model. Although its execution is currently commented out in `main.py`, it is fully functional and provides a robust way to calculate accuracy metrics on a test set.