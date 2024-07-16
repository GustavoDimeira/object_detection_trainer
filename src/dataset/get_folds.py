import argparse, random, os, json, funcy, shutil
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()

foldsHelp = "Define on how many folds your dataset will be splited"
valPercHelp = "Define the percentadge of images that will be used on validation"

parser.add_argument('--folds', type=int, default=5, help=foldsHelp)
parser.add_argument('--valPerc', type=float, default=.3, help=valPercHelp)

args = parser.parse_args()
folds = args.folds
valPerc = args.valPerc

dir_root = os.path.dirname(__file__)

O_CocoJSON = COCO(os.path.join(dir_root, "all/train/_annotations.coco.json"))
N_CocoJSON = COCO(os.path.join(dir_root, "all/newDataSet/_annotations.coco.json"))

O_images = O_CocoJSON.imgs
N_images = N_CocoJSON.imgs

info = O_CocoJSON.dataset["info"]
licenses = O_CocoJSON.dataset["licenses"]
categories = O_CocoJSON.dataset["categories"]
annotations = O_CocoJSON.dataset["annotations"]

O_fileNames = [O_images[i]["file_name"].split(".")[0] for i in range(len(O_images.keys()))]

O_totalImages = len(O_images)
testPerFolds = O_totalImages // folds
rest = O_totalImages % folds

random.shuffle(O_fileNames)

O_test = []; O_trainVal = []
N_test = []; N_trainVal = []

def save_coco(file, info, licenses, images, annotations, categories):
    with open(file, 'wt', encoding='UTF-8') as coco:
        json.dump({ 'info': info, 'licenses': licenses, 'images': images, 
            'annotations': annotations, 'categories': categories}, coco, indent=2)


def filter_annotations(annotations, images):
    image_ids = funcy.lmap(lambda i: int(i['id']), images)
    return funcy.lfilter(lambda a: int(a['image_id']) in image_ids, annotations)


if (os.path.exists(os.path.join(dir_root, "folds"))):
    shutil.rmtree(os.path.join(dir_root, "folds"))

os.mkdir(os.path.join(dir_root, "folds"))
os.mkdir(os.path.join(dir_root, "folds/new"))
os.mkdir(os.path.join(dir_root, "folds/original"))

for i in range(folds):
    if i < rest:
        test_start = i * (testPerFolds + 1)
        test_end = test_start + (testPerFolds + 1)
        namesImgTest = O_fileNames[test_start:test_end]
    else:
        test_start = rest * (testPerFolds + 1) + (i - rest) * testPerFolds
        test_end = test_start + testPerFolds
        namesImgTest = O_fileNames[test_start:test_end]

    for N_img in N_images:
        if N_images[N_img]["file_name"].split("&")[0] in namesImgTest:
            N_test.append(N_images[N_img])
        else:
            N_trainVal.append(N_images[N_img])

    for O_img in O_images:
        if O_images[O_img]["file_name"].split(".")[0] in namesImgTest:
            O_test.append(O_images[O_img])
        else:
            O_trainVal.append(O_images[O_img])

    O_trainFile = os.path.join(dir_root, f"folds/original/fold_{i+1}_train.json")
    O_valFile = os.path.join(dir_root, f"folds/original/fold_{i+1}_val.json")
    O_TestFile = os.path.join(dir_root, f"folds/original/fold_{i+1}_test.json")

    N_trainFile = os.path.join(dir_root, f"folds/new/fold_{i+1}_train.json")
    N_valFile = os.path.join(dir_root, f"folds/new/fold_{i+1}_val.json")
    N_TestFile = os.path.join(dir_root, f"folds/new/fold_{i+1}_test.json")

    x, y = train_test_split(O_trainVal, test_size=valPerc)
    x, y = train_test_split(N_trainVal, test_size=valPerc)

    save_coco(O_trainFile, info, licenses, x, filter_annotations(annotations, x), categories)
    save_coco(O_valFile, info, licenses, y, filter_annotations(annotations, y), categories)
    save_coco(O_TestFile, info, licenses, O_test, filter_annotations(annotations, O_test), categories)

    save_coco(N_trainFile, info, licenses, x, filter_annotations(annotations, x), categories)
    save_coco(N_valFile, info, licenses, y, filter_annotations(annotations, y), categories)
    save_coco(N_TestFile, info, licenses, N_test, filter_annotations(annotations, N_test), categories)
