import os, math, shutil, json, copy, numpy as np
from PIL import Image
from pycocotools.coco import COCO

ann_path = "all/train/_annotations.coco.json"
img_root = "all/train"
saving_path = "all/newDataSet"

def tilling(tile_size):
    # recreate the saving directory;
    if (os.path.exists("all/newDataSet")):
        shutil.rmtree("all/newDataSet")
        
    os.mkdir("all/newDataSet")

    coco = COCO(ann_path) # load annotations;

    new_coco_data = {
        'info': coco.dataset['info'],
        'licenses': coco.dataset['licenses'],
        'categories,': coco.dataset['categories'],
        'images': [],
        'annotations': [],
    } # will be used to save the new annotations afther the dataset aumentation;

    # iterate over every image, spliting then in square tiles and removing cutted ann;
    for i in range(len(coco.imgs)):
        img = coco.imgs[i]
        anns = coco.imgToAnns[i]
        img_name = img["file_name"]

        # converte img into numpy tensor;
        img_tensor = np.array(Image.open(os.path.join(img_root, img_name)))

        height, width, _ = img_tensor.shape
        # define the amount of cutts based on the new img with & height;
        y_cutts = math.ceil(height / tile_size)
        x_cutts = math.ceil(width / tile_size)

        new_height = height // y_cutts
        new_width = width // x_cutts

        print(f"Imagem {img_name.split('.')[0]} repartia em {x_cutts} por {y_cutts}.")

        # iterate over the annotations to remove those that are overlapping with the cutts;
        for ann in anns:
            x1, y1, w, h = ann["bbox"]
            x2, y2 = [int(w + x1), int(h + y1)]

            x1_tile = x1 // new_width
            x2_tile = x2 // new_width

            y1_tile = y1 // new_height
            y2_tile = y2 // new_height

            # add to the ann list if both "x" and "y" are in the same tile;
            if (x1_tile == x2_tile and y1_tile == y2_tile):
                tile_pos = x1_tile * x_cutts + y1_tile
                ann['id'] = len(new_coco_data['annotations'])
                ann['image_id'] = len(new_coco_data['images']) + tile_pos

                new_coco_data['annotations'].append(ann)
            else: # if the ann is in multiple tiles remove from the image;
                img_tensor[y1 : y2, x1 : x2, :] = [0, 0, 0]

        for x in range(x_cutts):
            for y in range(y_cutts):
                # cutt the image in tiles;
                img_tile_tensor = img_tensor[
                    new_height * y : new_height * (y +1),
                    new_width * x : new_width * (x +1),
                    :]
                
                # add the img info to the _annotation file;
                new_img_infos = copy.deepcopy(img)

                new_img_infos['id'] = len(new_coco_data['images'])
                new_img_infos['width'] = new_width
                new_img_infos['height'] = new_height
                new_img_infos['file_name'] = f"{img_name.split('.')[0]}_{x}_{y}.jpg"
                
                new_coco_data['images'].append(new_img_infos)

                new_img = Image.fromarray(img_tile_tensor) # save the tile as a new img;
                new_img.save(os.path.join(saving_path, new_img_infos['file_name']))

    # write the new annotation;
    with open(os.path.join(saving_path, "_annotations.json"), 'w') as json_file:
        json.dump(new_coco_data, json_file, indent=4)
