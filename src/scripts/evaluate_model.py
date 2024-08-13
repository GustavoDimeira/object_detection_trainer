import cv2, torch, numpy as np, os, sys, math
from mmdet.apis import inference_detector, init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.coco import CocoDataset
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from experiment_config.hyper_parameters import LIMIAR_IOU

ol_perc = .05

def is_max_score_thr(bb1, pred_array):
  """
    Compares if given bounding box is the one with the highest score_thr inside the array of predicted bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    pred_array : array of predicted objects
        Keys of dicts: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    boolean
  """
  is_max = True
  for cls in pred_array:
    for bb2 in cls:
      bbd={'x1':int(bb2[0]),'x2':int(bb2[2]),'y1':int(bb2[1]),'y2':int(bb2[3])}
      if is_max and bb2[4] > bb1['score_thr'] and get_iou(bb1,bbd) > LIMIAR_IOU:
        is_max = False
  return is_max

# IOU 
def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2', 'score_thr'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # print(bb1)
    # print(bb2)
    if bb1['x1'] >= bb1['x2'] or bb1['y1'] >= bb1['y2'] or bb2['x1'] >= bb2['x2'] or bb2['y1'] >= bb2['y2']:
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
#   print("iou:",str(iou))
    return iou


def segment_image(image, max_tile_size, ol_perc):
    height, width = image.shape[:2]
    print(height, width)
    
    # Calcular o número de tiles necessários para cobrir a imagem
    num_tiles_x = math.ceil(width / max_tile_size)
    num_tiles_y = math.ceil(height / max_tile_size)
    
    # Calcular o tamanho dos tiles
    tile_width = width // num_tiles_x
    tile_height = height // num_tiles_y
    
    tiles = []

    y_pos = 0; x_pos = 0
    for y in range(0, height, tile_height):  
        for x in range(0, width, tile_width):
            x_start = int(x - (0 if x == 0 else tile_width * ol_perc))
            x_end = int(x + tile_width + (0 if x == tile_width else tile_width * ol_perc))

            y_start = int(y - (0 if y == 0 else tile_height * ol_perc))
            y_end = int(y + tile_height + (0 if y == tile_height else tile_height * ol_perc))

            tiles.append([
                image[y_start:y_end, x_start:x_end, :],
                x_pos,
                y_pos
            ])
            x_pos += 1
        x_pos = 0
        y_pos += 1
    
    return (tiles, tile_width, tile_height)


def filter_result(predictions, x_size = 0, y_size = 0, ol_perc = 0):
    return predictions
    pass


def evaluate_model(cfg, pth_path):
    # Configurações do modelo
    cfg.data.test.test_mode = True
    torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None

    modelx = init_detector(cfg, pth_path)

    # Carregar e processar o dataset
    ann_file = cfg.data.test.ann_file
    img_prefix = cfg.data.test.img_prefix
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    coco_dataset = CocoDataset(
        ann_file=ann_file,
        classes=cfg.classes,
        data_root=cfg.data_root,
        img_prefix=img_prefix,
        pipeline=cfg.data.test.pipeline,
        filter_empty_gt=False
    )

    results = []
    medidos = []
    preditos = []
    all_TP = 0
    all_FP = 0
    all_GT = 0

    # iterar sobre todas as imagens do fold
    for index, info in enumerate(coco_dataset.data_infos):
        file_name = info['file_name']
        img_path = os.path.join("results", img_prefix, file_name)
        image = cv2.imread(img_path)

        (tiles, tile_width, tile_height) = segment_image(image, 1280, ol_perc)

        final_result = []
        for [tile, x_pos, y_pos] in tiles:
            tile_result = inference_detector(modelx, tile)

            for class_id in range(len(tile_result)):
                for i, prediction in enumerate(tile_result[class_id]):
                    tile_result[class_id][i] = {
                        "x1": prediction["x1"] + (tile_width * max((x_pos - ol_perc), 0)),
                        "x2": prediction["x2"] + (tile_width * max((x_pos - ol_perc), 0)),
                        "y1": prediction["y1"] + (tile_height * max((y_pos - ol_perc), 0)),
                        "y2": prediction["y2"] + (tile_height * max((y_pos - ol_perc), 0)),
                        "score_thr": prediction["score_thr"]
                    }

            final_result.append(tile_result)

        final_result = filter_result(final_result)

        # Comparar as predições com as anotações (Ground Truth)
        gt_annotations = coco_dataset.get_ann_info(index)
        gt_bboxes = gt_annotations['bboxes']
        gt_labels = gt_annotations['labels']

        TP = 0
        FP = 0

        for class_id in range(len(cfg.classes)):
            pred_bboxes = np.array([res for res in final_result[class_id] if res['class'] == class_id])
            gt_bboxes_class = np.array([gt_bboxes[i] for i in range(len(gt_labels)) if gt_labels[i] == class_id])

            medidos.append(len(gt_bboxes_class))
            preditos.append(len(pred_bboxes))

            all_GT += len(gt_bboxes_class)

            # Comparar predições com ground truth
            for pred_bbox in pred_bboxes:
                ious = [get_iou(pred_bbox, gt_bbox) for gt_bbox in gt_bboxes_class]
                max_iou = max(ious) if ious else 0
                if max_iou >= 0.5:  # Se IoU for maior que 0.5, considerar como TP
                    TP += 1
                    all_TP += 1
                else:
                    FP += 1
                    all_FP += 1

        results.append({
            'file_name': file_name,
            'TP': TP,
            'FP': FP,
            'GT': len(gt_bboxes)
        })

    return results, medidos, preditos, all_TP, all_FP, all_GT
