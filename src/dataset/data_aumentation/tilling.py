import os, math, shutil, json, copy, numpy as np, sys
from PIL import Image
from pycocotools.coco import COCO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.utils.terminal_handler import LoadingBar

dir_root = os.path.join(os.path.dirname(__file__), "../all")

ann_path = os.path.join(dir_root, "train/_annotations.coco.json")
img_root = os.path.join(dir_root, "train")
saving_path = os.path.join(dir_root, "newDataSet")

def tilling(tile_size):
    # recreate the saving directory;
    if (os.path.exists(os.path.join(dir_root, "newDataSet"))):
        shutil.rmtree(os.path.join(dir_root, "newDataSet"))
        
    os.mkdir(os.path.join(dir_root, "newDataSet"))

    coco = COCO(ann_path) # load annotations;

    new_coco_data = {
        'info': coco.dataset['info'],
        'licenses': coco.dataset['licenses'],
        'categories,': coco.dataset['categories'],
        'images': [],
        'annotations': [],
    } # will be used to save the new annotations afther the dataset aumentation;

    amm_imgs = 0 # counter
    total_images = len(coco.imgs)

    bar = LoadingBar(total_images, "Spliting images", ["", ""])
    bar.start()

    # iterate over every image, spliting then in square tiles and removing cutted ann;
    for i in range(total_images):
        colluns, _ = shutil.get_terminal_size()
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

        imgSplitMsg = f"Imagem {img_name.split('.')[0]} repartia em {x_cutts} por {y_cutts}."
        imgSplitMsg += "-" * (colluns - len(imgSplitMsg) - 1) + "|"

        amm_imgs += x_cutts * y_cutts

        bar.extraInfos = [
            f"Processing image: {img_name.split('.')[0]}.",
            f"{amm_imgs} images generate so far."
        ]
        bar.updateBar(i  + 1)

        # iterate over the annotations to remove those that are overlapping with the cutts;
        for ann in anns:
            x1, y1, w, h = ann["bbox"]
            x2, y2 = [int(w + x1), int(h + y1)]

            x1_tile = x1 // new_width
            x2_tile = x2 // new_width

            y1_tile = y1 // new_height
            y2_tile = y2 // new_height

            # add to the ann list if both "x" and "y" are at the same tile;
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
                new_img_infos['file_name'] = f"{img_name.split('.')[0]}&{x}_{y}.jpg"
                
                new_coco_data['images'].append(new_img_infos)

                new_img = Image.fromarray(img_tile_tensor) # save the tile as a new img;
                new_img.save(os.path.join(saving_path, new_img_infos['file_name']))

    final_msg = f"{amm_imgs - total_images} images were created over the inicial {total_images}, that is a {(amm_imgs / total_images * 100 - 100):.2f}% increase!!"
    final_msg = " " * ((colluns - len(final_msg)) // 2) + final_msg

    print(colluns * "=")
    print(final_msg)
    print(colluns * "=")

    # write the new annotation;
    with open(os.path.join(saving_path, "_annotations.coco.json"), 'w') as json_file:
        json.dump(new_coco_data, json_file, indent=4)


if (__name__ == "__main__"):
    tilling(1000)