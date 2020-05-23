import argparse
import logging

from libs.utils.logger import setupLogger
from libs.utils.tfutils import TFEvaluator
from libs.utils.torchutils import TorchEvaluator

def setup_parser():
    parser = argparse.ArgumentParser(description='Networks tests')

    parser.add_argument('--model', type=str, default='fcn',
    choices = ['fcn', 'deeplabv3', 'deeplabv3plus'], help = 'model name')
    parser.add_argument('--datapath', default='./datasets/my/', help = 'dataset path')
    #parser.add_argument('--listname', type=str, default='imagelist.txt', help = 'name of file with list of images')
    parser.add_argument('--gpu', action = 'store_true', default=False, help = 'Enable GPU for Torch models')

    return parser

if __name__ == '__main__':

    parser = setup_parser()
    args = parser.parse_args()

    logger = setupLogger("TEST_" + args.model, "./logs/")
    logger.info(f"Model is {args.model}, dataset path is {args.datapath}")

    if args.model == "deeplabv3plus":
        evaluator = TFEvaluator(args.model, args.datapath)
    else:
        evaluator = TorchEvaluator(args.model, args.datapath, args.gpu)

    logger.info(f"Starting evaluating...")
    evaluator.eval()

    """
    input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
    d = TorchDataLoader(root = "./datasets/VOCtest", transform = input_transform)
    sampler = data.sampler.SequentialSampler(d)
    batch_sampler = data.sampler.BatchSampler(sampler, 1, drop_last=True)
    dl = data.DataLoader(dataset=d, batch_sampler = batch_sampler, num_workers = 4, pin_memory = True)
    
    model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)
    model.to('cuda')
    model.eval()

    metric = Metric(21)

    for i, (image, mask) in enumerate(dl):
        image = image.to('cuda')
        mask = mask.to('cuda')
        with torch.no_grad():
            outputs = model(image)

        metric.update(outputs['out'], mask)
        pixAcc, mIoU = metric.get()
        print(f"Sample {i+1}, pixAcc: {pixAcc*100}, mIoU: {mIoU*100}")
    
    """
    """
    ds = TFDataSetMy("datasets/")
    DL = TFDataLoader(ds = ds)

    #d = TFDataLoader(root = "./datasets/")
    #sampler = data.sampler.SequentialSampler(d)
    #batch_sampler = data.sampler.BatchSampler(sampler, 1, drop_last=True)
    #DL = data.DataLoader(dataset=d, batch_sampler = batch_sampler, num_workers = 4, pin_memory = True)
    
    model = Deeplabv3(backbone='xception', OS=8)
    metric = TFMetric(21)

    for i, (image, mask) in enumerate(DL):

        #print(image, mask)
        #print(type(image))

        pred = model.predict(image)
        res = np.argmax(pred.squeeze(), -1)

        #print(target)
        #print(res)
        metric.update(res, mask)
        pixAcc, mIoU = metric.get()
        print(f"Sample {i+1}, pixAcc: {pixAcc*100}, mIoU: {mIoU*100}")
        
    """
    #model = torch.hub.load('pytorch/vision:v0.6.0', 'deeplabv3_resnet101', pretrained=True)

    #model = Deeplabv3(backbone='xception', OS=8)

    #print(torch.cuda.is_available())
    #p = Predictor(model, "torch", gpu = torch.cuda.is_available())

    #p.process_image("./1.jpg", "./")
    #video_path = './data/2.mp4'
    #p.process_video(video_path, 'res.avi')