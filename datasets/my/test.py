from pathlib import Path
from PIL import Image
import numpy as np

with open("imagelist.txt", 'r') as lines:
    for line in lines:
        # iname = (line.rstrip('\n') + ".jpg")
        mname = (line.rstrip('\n') + ".png")
        mask = Image.open("mask/" + mname)

        mask = np.array(mask)

        mask[mask != 15] = 0

        res = Image.fromarray(mask)
        res.save("res/" + mname)

        print(np.unique(mask))

        