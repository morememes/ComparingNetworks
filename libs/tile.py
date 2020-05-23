import numpy as np

class Tile:

    def __init__(self, roi:np.ndarray,
                 frame_n:int, rate:float,
                 left:int, up:int):

        self.roi = roi
        self.frame_n = frame_n
        self.rate = rate
        self.left = left
        self.up = up
        self.mask = None

        self.im_info = None

    def __str__(self):
        return f'frame({self.frame_n: >5}), rate({self.rate: >6}), left({self.left: >5}), up({self.up: >5})'

    @staticmethod
    def split_frame(frame:np.ndarray,
                    frame_n:int,
                    rate:float=1,
                    gap:int=100,
                    subsize:int=512):

        slide = subsize - gap

        if rate != 1:
            interpolation = cv2.INTER_CUBIC if rate > 1.0 else cv2.INTER_AREA
            frame = cv2.resize(frame, None, fx=rate, fy=rate, interpolation=interpolation)

        height, width = frame.shape[:2]

        tiles = []

        left, up = 0, 0
        right, bottom = None, None
        while left < width:
            if left + subsize >= width:
                left = max(width - subsize, 0)
            up = 0
            while up < height:
                if up + subsize >= height:
                    up = max(height - subsize, 0)

                right = left + subsize
                bottom = up + subsize

                roi = frame[up:bottom, left:right].copy()
                tile = Tile(roi, frame_n, rate, left, up)
                tiles.append(tile)

                if up + subsize >= height:
                    break
                else:
                    up = up + slide
            if left + subsize >= width:
                break
            else:
                left = left + slide

        return tiles