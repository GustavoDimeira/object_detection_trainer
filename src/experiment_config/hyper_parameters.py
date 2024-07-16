import os, json
from ..utils.models_base_config import models_dict

# -----------------------
# Define hyperparameters
# -----------------------

EPOCHS = 100
PATIENCE = .05
LIMIAR_CLASSIFICADOR = 0.5
LIMIAR_IOU = 0.5
EARLY_STOP = True

# -----------------------
# Define other variabels
# -----------------------

SAVE_IMAGES = False
SKIP_TRAIN = True
OVERPASS_EXCEPTIONS = True

# -------------------------------------------
# Define the architectures, optmizers and lr
# -------------------------------------------

MODELS = {
    'sabl': models_dict.sabl.sabl_retinanet_r50_fpn_1x_coco,
    'fovea': models_dict.foveabox.fovea_r50_fpn_4x4_1x_coco,
    'faster': models_dict.faster_rcnn.faster_rcnn_r50_fpn_1x_coco,
    'retinanet': models_dict.retinanet.retinanet_r50_fpn_1x_coco,
    'atss': models_dict.atss.atss_r50_fpn_1x_coco,
}

OPTIMIZERS = ["SGD", "AdamW"]
LEARNING_RATES = [10 ** -2, 10 ** -3, 10 ** -4]

# ----------------------------------------
# Automated definition of a few variables
# ----------------------------------------

CLASSES = ()
with open('../dataset/annotation/train/_annotations.coco.json', 'r') as file:
    data = json.load(file)
    ann_ids = []

    for anotation in data["annotations"]:
        if anotation["category_id"] not in ann_ids:
            ann_ids.append(anotation["category_id"])

    for category in data["categories"]:
        if category["id"] in ann_ids:
            CLASSES += (category["name"],)

FOLDS = len(os.listdir(r'../dataset/folds')) // 3 # there are 3 files for each fold
