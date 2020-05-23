import numpy as np
import cv2
import torch
import logging
from torchvision import transforms
from libs.tile import Tile
from libs.deeplab.model import preprocessinput

from pathlib import Path

class Predictor:
    def __init__(self, model, mode, gpu = False):
        self.model = model
        self.mode = mode
        self.gpu = gpu

    def __make_preds__(self, pic):
        """
        Метод принимает на вход батч, возвращает маски для батча.
        """
        w, h, _ = pic.shape
        if self.mode == "torch":
            if min(w, h) < 600:
                res = self.__make_preds_torch__(pic)
            else:
                res = self.__make_preds_torch_with_tiles__(pic)

        if self.mode == "tf":
            if min(w, h) < 512:
                res = self.__make_preds_tf__(pic)
            else:
                res = self.__make_preds_tf_with_tiles__(pic)

        res[ res != 15 ] = 0 
        return res

    def __make_preds_torch__(self, pic):
        if self.gpu:
            self.model.to('cuda')
        self.model.eval()

        # Создаем предпроцессинг
        preprocessinput = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        input_tensor = preprocessinput(pic)
        batch_in = input_tensor.unsqueeze(0)
        if self.gpu:
            batch_in = batch_in.to('cuda')

        with torch.no_grad():
            output = self.model(batch_in)['out'].cpu()
        res = np.argmax(output.squeeze(), 0)

        res = res.numpy().astype(np.uint8)

        return res
  
    def __make_preds_torch_with_tiles__(self, pic):
        # Сохраняем высоту, ширину модели
        w, h, _ = pic.shape

        # Подготавливаем модель
        if self.gpu:
            self.model.to('cuda')
        self.model.eval()

        # Разбиваем изображение на тайлы
        tiles = Tile.split_frame(pic, 0)

        tiles_len = len(tiles)

        # Создаем предпроцессинг
        preprocessinput = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

        # Подготавливаем данные на вход сети
        tiles_res = [preprocessinput(tile.roi).unsqueeze(0) for tile in tiles]
        batch_in = torch.from_numpy(np.concatenate(tiles_res, axis=0))
        
        if self.gpu:
            batch_in = batch_in.to('cuda')

        #------------------------------------------------------

        batch_size = min(tiles_len, 3)

        # Получаем предсказания

        with torch.no_grad():
            results = [self.model(batch_in[k: min(batch_size, tiles_len - k) + k])['out'].cpu() for k in range(0, tiles_len, batch_size)]
        batch_out = np.concatenate(results, axis=0)

        res = np.argmax(batch_out.squeeze(), 1)

        #------------------------------------------------------

        shape = (w, h)
        mask = np.zeros(shape, dtype=np.uint8)

        for k in range(len(tiles)):
            mask[tiles[k].up : tiles[k].up + 512,tiles[k].left : tiles[k].left + 512] = res[k]

        mask = mask.astype(np.uint8)

        return mask

    def __make_preds_tf__(self, pic):
        w, h, _ = pic.shape
        print(w, h)
        img = cv2.resize(pic, (512, 512))

        cv2.imwrite("test.jpg", img)

        batch_in = preprocessinput(img[None])
        res = self.model.predict(batch_in)
        res = np.argmax(res.squeeze(), -1).astype(np.uint8)

        res_resized = cv2.resize(res, (h, w))
        return np.array(res_resized, dtype=np.uint8)


    def __make_preds_tf_with_tiles__(self, pic):
        # Сохраняем высоту, ширину модели

        w, h, _ = pic.shape
    
        tiles = Tile.split_frame(pic, 0)
        tiles_len = len(tiles)

        tiles_res = [tile.roi[None] for tile in tiles]
        batch_in = np.concatenate(tiles_res, axis=0)

        batch_in = preprocessinput(batch_in)

        batch_size = min(tiles_len, 3)
        results = np.array([self.model.predict(batch_in[k: min(batch_size, tiles_len - k) + k]) for k in range(0, tiles_len, batch_size)], dtype=np.uint8)
        results = np.concatenate(results, axis=0)
        
        res = np.argmax(results.squeeze(), -1)

        shape = (w, h)
        mask = np.zeros(shape, dtype=np.uint8)


        for k in range(len(tiles)):
            mask[tiles[k].up : tiles[k].up + 512,tiles[k].left : tiles[k].left + 512] = res[k]

        mask = mask.astype(np.uint8)

        #for i in range(w):
        #    for k in range(h):
        #        if mask[i,k] != 15:
        #            mask[i, k] = 0

        #mask[ mask != 15 ] = 0

        return mask


    # Пользовательские методы, для работы с изображениями и видео на прямую
    # Зависят от методов выше

    def process_image(self, image_path, save_path):
        image_path = Path(image_path)
        save_path = Path(save_path)        
        if save_path.is_dir():
            res_path = save_path / (image_path.stem + "_res" + image_path.suffix)
            save_path = save_path / (image_path.stem + "_mask.png" )

        pic = cv2.imread(str(image_path))
        mask = self.__make_preds__(pic)
        cv2.imwrite(str(save_path), mask)

        img = cv2.bitwise_and(pic, pic, mask=mask)
        cv2.imwrite(str(res_path), img)
        


    def process_video(self, video_path, save_path, newfps = None):

        logger = logging.getLogger("MAIN.process_video")
        
        reader = cv2.VideoCapture(video_path)
        frame_amount = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(reader.get(3)), int(reader.get(4))
        size = (w, h)
        ifps = float(reader.get(5))
        fps = ifps
        if newfps is not None and newfps < fps:
            ratio = int(round(fps) // newfps)
            fps = float(round(fps) // ratio)

        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)

        logger.info(f"Processing {ifps} fps video with {w}x{h} resolution, newfps is {fps}")
        logger.info(f"TOTAL FRAMES FOR PROCESSING: {frame_amount}")

        i = 0
        ret = True
        while ret:
            ret, frame = reader.read()
            if i % ratio == 0:
                logger.info(f"Processing {i} frame...")
                if ret:
                    frame = frame.astype(np.uint8)
                    mask = self.__make_preds__(frame)
                    img = cv2.bitwise_and(frame, frame, mask=mask)
                    writer.write(img)
            i += 1

        reader.release()
        writer.release()


    def process_video_rt(self):
        pass
