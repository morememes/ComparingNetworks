import logging
from pathlib import Path
import numpy as np
from PIL import Image

import tensorflow as tf
from libs.deeplab.model import preprocessinput
from libs.deeplab.model import Deeplabv3


class TFDataSetMy(object):
    
    def __init__(self, root):
        super(TFDataSetMy, self).__init__()

        self.root = Path(root)
        
        f_list = self.root / "imagelist.txt"
        image_dir = self.root / "img"
        mask_dir = self.root / "mask"

        assert (f_list.exists() == True)

        self.images = []
        self.masks = []

        with open(f_list, "r") as lines:
            for line in lines:
                path = image_dir / (line.rstrip('\n') + ".jpg")
                if path.exists():
                    self.images.append(str(path))

                path = mask_dir / (line.rstrip('\n') + ".png")
                if path.exists():
                    self.masks.append(str(path))

        assert (len(self.images) == len(self.masks))

        print(f"Found {len(self.images)} images in the folder")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        img, mask = self._img_transform(img), self._mask_transform(mask)

        return preprocessinput(img[None]), mask

    def __len__(self):
        return len(self.images)

    def _img_transform(self, img):
        img = img.resize((512, 512))
        img = np.array(img).astype('int32') 
        return img

    def _mask_transform(self, mask):
        mask = mask.resize((512, 512))
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return target

class TFDataLoader(object):
    def __init__(self, ds):
        super(TFDataLoader, self).__init__()
        self.ds = ds
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < len(self.ds):
            result_id = self.index
            self.index += 1
            return self.ds.__getitem__(result_id)
        raise StopIteration


class TFMetric(object):
    def __init__(self, nclass):
        super(TFMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        def evaluate_worker(self, pred, label):
            correct, labeled = pixelAccuracy(pred, label)
            inter, union = intersectionAndUnion(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, np.ndarray):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.zeros(self.nclass)
        self.total_union = np.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0

def pixelAccuracy(imPred, imLab):

    pixel_labeled = np.sum(imLab >= 0)
    pixel_correct = np.sum((imPred == imLab) * (imLab >= 0))
    return (pixel_correct, pixel_labeled)

def intersectionAndUnion(imPred, imLab, numClass):

    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    #print(np.unique(intersection))
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)

def get_model_tf(model_name):
    models = {
        "deeplabv3plus" : Deeplabv3(backbone='xception', OS=8)
    }
    return models[model_name]

def get_loader_tf(dataset_path):
    ds = TFDataSetMy(dataset_path)
    DL = TFDataLoader(ds = ds)
    return DL

class TFEvaluator:
    def __init__(self, model_name, dataset_path):
        self.model = get_model_tf(model_name)
        self.dataloader = get_loader_tf(dataset_path)
        self.metric = TFMetric(21)
        self.logger = logging.getLogger("TEST_"+model_name+".Evaluator")

    def eval(self):
        for i, (image, target) in enumerate(self.dataloader):
            pred = self.model.predict(image)
            res = np.argmax(pred.squeeze(), -1)
            self.metric.update(res, target)
            
            pixAcc, mIoU = self.metric.get()
            self.logger.info(f"Sample {i+1}, pixAcc: {pixAcc*100}, mIoU: {mIoU*100}")
