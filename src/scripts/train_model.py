import torch, time, mmcv, os.path as osp

from mmdet import __version__

from mmdet.datasets import build_dataset
from mmdet.apis import train_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.models import build_detector

from mmcv.utils import get_git_hash

def train_model(cfg, model_name):
    print('Create workdir:', mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir)))

    torch.backends.cudnn.benchmark = True
    distributed = False

    cfg.workflow = [('train', 1), ('val', 1)]
    cfg.dump(osp.join(cfg.work_dir, osp.basename(model_name+'.py')))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    meta = dict()

    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = cfg.seed
    meta['exp_name'] = osp.basename(model_name+'.py')

    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    datasets = [build_dataset(cfg.data.train, dict(test_mode=False, filter_empty_gt=False))]
    datasets.append(build_dataset(cfg.data.val, dict(test_mode=False, filter_empty_gt=False)))

    datasets[0].CLASSES = cfg.classes
    datasets[1].CLASSES = cfg.classes

    cfg.checkpoint_config.meta = dict(
        mmdet_version=__version__ + get_git_hash()[:7],
        CLASSES=datasets[0].CLASSES)

    model = build_detector(cfg.model, train_cfg=cfg.get(
        'train_cfg'), test_cfg=cfg.get('test_cfg'))

    model.CLASSES = datasets[0].CLASSES

    try:
        train_detector(model, datasets, cfg, distributed=False,
                    validate=False, timestamp=timestamp, meta=meta)
    except:
        print("Traing forced stopped by not impoving")