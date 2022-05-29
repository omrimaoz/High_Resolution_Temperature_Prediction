import sys

import numpy as np
import tifffile as tiff
from PIL import Image

path = 'Zeelim_29.5.19_1730_W/PredictedIR.tif'

image = tiff.imread('./resources/{path}'.format(path=path))
image = image - np.min(image)
image = (image / np.max(image)) * 255
image = image.astype(np.uint8)
Image.fromarray(image).show()
print(1)