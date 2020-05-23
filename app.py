#import torch
import tensorflow as tf

from libs.utils.logger import setupLogger
from libs.core import Predictor

from libs.deeplab.model import Deeplabv3

import time

if __name__ == '__main__':
    logger = setupLogger("MAIN", "./logs/")
    logger.info("Starting main function")


    #model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
    model = Deeplabv3(backbone='xception', OS=8)
    
    logger.info("Model is loaded")

    p = Predictor(model, "tf")#, gpu = torch.cuda.is_available())

    logger.info("Predictor is created")

    """
    logger.info("Starting process image")
    save_dir = "./samples/plus"
    p.process_image("./samples/2007_000170.jpg", save_dir)
    logger.info("Processing image is finished")
    """

    video_path = './data/video/2.mp4'
    logger.info("Starting process image")

    start = time.time()
    p.process_video(video_path, 'res.avi', 30.0)
    end = time.time()
    logger.info(f"Processing image is finished with {round(float(end - start), 2)} seconds")