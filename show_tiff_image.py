import sys

import numpy as np
import tifffile as tiff
from PIL import Image

path = 'Mishmar_3.3.20_0910_W/dynamic_output_raster.tif'

image = tiff.imread('./resources/{path}'.format(path=path))
image = image - np.min(image)
image = (image / np.max(image)) * 255
image = image.astype(np.int8)
Image.fromarray(image).show()
print(1)