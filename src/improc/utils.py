import numpy as np
import matplotlib.pyplot as mtpplt
import os

def split_img_channels(img_path : str, channel_count : int = 0) -> list :
    """
    This function will split the channels of the image
    
    **Args:**
        img_path - path of image.
        channel_count - number of channels to extract as list.

    By default channel_count = 0 (i.e all available) & the order is R, G, B 

    Returns:
        list of channel seperated images
    """

    if not os.path.isfile(img_path):
        raise Exception("Invalid File path")
    
    img = mtpplt.imread(img_path)
    # print(img)

    if channel_count > img.ndim:
        raise Exception("Cannot extract more channel than available in Image")
    elif channel_count < 0:
        raise Exception("Channel count should be non-negative and non-zero")
    elif channel_count == 0:
        channel_count = img.ndim

    channel_list = []

    for i in range(channel_count):
        cha_val = np.zeros_like(img) 
        cha_val[:, :, i] = img[:, :, i]
        channel_list.append(cha_val)
        # print( " This is temp: ", i, cha_val)
        # print("Channel data", img[:,:,i])

    return channel_list


def pixel_iterator(img_src : np.ndarray, callback: any, img_dest : np.ndarray) -> None:
    """
    This function iterate over the image pixel by pixel providing 
    callback to be called on that pixel and reassign on dest

    Args:
        img_src - ndarray image source
        callback - callback function taking pixel (R, G, B) as arguemtn & return pixel
        img_dest - ndarray destination 

    Returns:
        None
    """

    if callback == None:
        raise Exception("Callback cannot be null")
    
    for x, row in enumerate(img_src):
        for y, col in enumerate(row):
            img_dest[x][y] = callback(col)


# 2d convolution
def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:

    img_width, img_height = img.shape[0], img.shape[1]
    k_width, k_height = kernel.shape
    pad_w = k_width // 2
    pad_h = k_height // 2

    padded_img = np.pad(img, ((pad_w, pad_w), (pad_h, pad_h)), mode = 'constant')

    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i: i + k_width, j: j + k_height]

            result[i, j] = np.sum(region * kernel)

    # np.clip(result, a_min = 0, a_max = 255)

    return result 


# 2d convolution
def conv2d_iter(img: np.ndarray, kernel: np.ndarray, op: any) -> np.ndarray:

    img_width, img_height = img.shape[0], img.shape[1]
    k_width, k_height = kernel.shape
    pad_w = k_width // 2
    pad_h = k_height // 2

    padded_img = np.pad(img, ((pad_w, pad_w), (pad_h, pad_h)), mode = 'constant')

    result = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded_img[i: i + k_width, j: j + k_height]

            result[i, j] = op(kernel, (np.sum(region * kernel)))

    # np.clip(result, a_min = 0, a_max = 255)

    return result 