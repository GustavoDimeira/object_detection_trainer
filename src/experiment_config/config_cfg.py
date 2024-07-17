import os, torch
from mmcv import Config
from mmdet.apis import set_random_seed

from utils.optmizers import optimizers
from experiment_config.hyper_parameters import MODELS, EARLY_STOP, CLASSES

def config_cfg(
    data_root,
    total_epochs,
    selected_model,
    learning_rate,
    fold,
    optimizer_type
):
    config_file = os.path.join("../mmdetection", MODELS[selected_model]['config_file'])
    fold = 'fold_' + str(fold)

    # create the cfg with the predefined model parameters
    cfg = Config.fromfile(config_file)

    cfg.data_root = data_root
    cfg.classes = CLASSES

    # defining configuration for test dataset
    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.ann_file = '../dataset/folds/original/'+fold+'_test.json' 
    cfg.data.test.classes = cfg.classes
    cfg.data.test.img_prefix = '../dataset/all/train'

    # defining configuration for train dataset
    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.ann_file = '../dataset/folds/new/'+fold+'_train.json'
    cfg.data.train.classes = cfg.classes
    cfg.data.train.img_prefix = '../dataset/all/newDataset'

    # defining configuration for val dataset
    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.ann_file = '../dataset/folds/new/'+fold+'_val.json'
    cfg.data.val.classes = cfg.classes
    cfg.data.val.img_prefix = '../dataset/all/newDataset'
    cfg.data.val.pipeline = cfg.data.train.pipeline

    if 'roi_head' in cfg.model:
        if not isinstance(cfg.model.roi_head.bbox_head, list):
            cfg.model.roi_head.bbox_head['num_classes'] = len(cfg.classes)
        else:
            for i in range(len(cfg.model.roi_head.bbox_head)):
                cfg.model.roi_head.bbox_head[i]['num_classes'] = len(cfg.classes)
    else:
        cfg.model.bbox_head['num_classes'] = len(cfg.classes)

    cfg.load_from = MODELS[selected_model]['checkpoint']

    hp_string = f'{selected_model}_{learning_rate}_{optimizer_type}'
    cfg.work_dir = os.path.join(data_root, fold, 'MModels', hp_string)

    # print('Modelos serão salvos aqui: ', cfg.work_dir)
    cfg.runner.max_epochs = total_epochs
    cfg.total_epochs = total_epochs

    cfg.optimizer = optimizers[optimizer_type]["optimizer"]
    cfg.optimizer.lr = learning_rate
    cfg.optimizer_config = optimizers[optimizer_type]["optimizer_config"]
    cfg.lr_config.policy = 'step'

    cfg.evaluation.metric = 'mAP'
    cfg.evaluation.save_best = 'auto'
    cfg.evaluation.interval = 1

    cfg.checkpoint_config.interval = 1 if EARLY_STOP else total_epochs
    cfg.checkpoint_config.create_symlink = True

    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"

    return cfg
