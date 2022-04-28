import PIL
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import tifffile as tiff
import scipy



a = tiff.imread("./resources/Zeelim_23.9.19_1100_E/height.tif")
img = scipy.misc.toimage(a)
np.array(img)

height_image = Image.open('./resources/Zeelim_23.9.19_1100_E/height.tif')
height_matrix = np.array(height_image)
normal_image = np.array(height_matrix / np.max(height_matrix) * 255, dtype=np.int)
image = Image.fromarray(normal_image)


print(1)



