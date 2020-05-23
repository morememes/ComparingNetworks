import logging
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import torch.utils.data as data

class TorchDataLoader(Dataset):
    #NUM_CLASS = 21

    def __init__(self, root='./datasets/', transform=None, base_size=520, crop_size=480):
        super(TorchDataLoader, self).__init__()

        self.root = Path(root)
        self.transform = transform
        #self.split = split
        #self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

        image_dir = self.root / "img"
        mask_dir = self.root / "mask"
        f_list = self.root / "imagelist.txt"

        
        self.images = []
        self.masks = []

        with open(f_list, "r") as lines:
            for line in lines:
                path = image_dir / (line.rstrip('\n') + ".jpg") 
                if path.exists():
                    self.images.append(path)

                path = mask_dir / (line.rstrip('\n') + ".png")
                if path.exists():
                    self.masks.append(path)

        assert (len(self.images) == len(self.masks))
        print(f"Found {len(self.images)} images in the folder")

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        mask = Image.open(self.masks[index])
        img, mask = self._img_transform(img), self._mask_transform(mask)

        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _img_transform(self, img):
        return np.array(img)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

class TorchMetric(object):

    def __init__(self, nclass):
        super(TorchMetric, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result.
        Parameters
        ----------
        labels : 'NumpyArray' or list of `NumpyArray`
            The labels of the data.
        preds : 'NumpyArray' or list of `NumpyArray`
            Predicted values.
        """

        def evaluate_worker(self, pred, label):
            correct, labeled = batch_pix_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)

            self.total_correct += correct
            self.total_label += labeled
            if self.total_inter.device != inter.device:
                self.total_inter = self.total_inter.to(inter.device)
                self.total_union = self.total_union.to(union.device)
            self.total_inter += inter
            self.total_union += union

        if isinstance(preds, torch.Tensor):
            evaluate_worker(self, preds, labels)
        elif isinstance(preds, (list, tuple)):
            for (pred, label) in zip(preds, labels):
                evaluate_worker(self, pred, label)

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pixAcc = 1.0 * self.total_correct / (2.220446049250313e-16 + self.total_label)  # remove np.spacing(1)
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        mIoU = IoU.mean().item()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = torch.zeros(self.nclass)
        self.total_union = torch.zeros(self.nclass)
        self.total_correct = 0
        self.total_label = 0


# pytorch version
def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1# remove np.spacing(1)) + 1
    target = target.long() + 1

    

    #print(np.unique(predict.cpu().numpy()))

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    predict[predict != 16] = 0
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()


def get_model_torch(model_name):
    if model_name == "fcn":
        model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
        return model
    if model_name == "deeplabv3": 
        model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
        return model


def get_loader_torch(dataset_path):
    input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
    d = TorchDataLoader(root = dataset_path, transform = input_transform)
    sampler = data.sampler.SequentialSampler(d)
    batch_sampler = data.sampler.BatchSampler(sampler, 1, drop_last=True)
    dl = data.DataLoader(dataset=d, batch_sampler = batch_sampler, num_workers = 4, pin_memory = True)

    return dl

class TorchEvaluator:
    def __init__(self, model_name, dataset_path, gpu = False):
        self.model = get_model_torch(model_name).eval()
        self.dataloader = get_loader_torch(dataset_path)
        self.metric = TorchMetric(21)
        self.gpu = gpu
        if self.gpu:
            self.model.to('cuda')

        self.logger = logging.getLogger("TEST_"+model_name+".Evaluator")

    def eval(self):

        for i, (image, target) in enumerate(self.dataloader):
            if self.gpu:
                image = image.to('cuda')
                target = target.to('cuda')
            with torch.no_grad():
                outputs = self.model(image)

            self.metric.update(outputs['out'], target)
            pixAcc, mIoU = self.metric.get()
            self.logger.info(f"Sample {i+1}, pixAcc: {pixAcc*100}, mIoU: {mIoU*100}")

