import random

import numpy as np
import PIL


def ShearX(img, v=0.1, fixed=True, random_mirror=True):
    if not fixed:
        v = np.random.uniform(low=0.0, high=v)
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, v=0.1, fixed=True, random_mirror=True):
    if not fixed:
        v = np.random.uniform(low=0.0, high=v)
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))


def Rotate(img, v=30, fixed=True, square=True, random_mirror=True):
    if not fixed:
        if square:
            v = random.choice([0, 90, 180, 270])
        else:
            v = np.random.uniform(low=0.0, high=v)
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)


def Equalize(img):
    return PIL.ImageOps.equalize(img)


def Cutout(img, v, color=(125, 123, 114)):
    if v < 0:
        return img

    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.0))
    y0 = int(max(0, y0 - v / 2.0))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    img_c = img.copy()
    PIL.ImageDraw.Draw(img_c).rectangle(xy, color)
    return img_c


def FlipUD(img):
    return PIL.ImageOps.flip(img)


def FlipLR(img):
    return PIL.ImageOps.mirror(img)


def Invert(img):
    return PIL.ImageOps.invert(img)


def Crop(img, crop_size=(4, 4)):
    img_array = np.array(img)
    w, h, _ = img_array.shape
    left = np.random.randint(0, w - crop_size[0])
    top = np.random.randint(0, h - crop_size[1])
    right = left + crop_size[0]
    bottom = top + crop_size[1]
    img_array = img_array[left:right, top:bottom, :]
    img = PIL.Image.fromarray(np.uint8(img_array))
    img = img.resize((w, h))
    return img


def PatchGaussian(img, patch_size=10, scale=0.2, fixed=True):
    """
    Args:
        img: the target image ([0, 255])
        patch_size: The size of a patch. The patch is square.
        scale: The Gaussian noise to apply.
        fixed: If False makes the uniformly at random makes mask size be between 1 and patch_size.
    """
    img_array = np.array(img) / 255.0

    if not fixed:
        patch_size = np.random.randint(1, patch_size + 1)
        # otherwise, patch_size is fixed.

    # apply path gaussian noise to image

    # # create a mask
    # ## randomly sample location in image:
    img_width, img_height, n_channels = img_array.shape
    x = np.random.randint(0, img_width + 1)
    y = np.random.randint(0, img_height + 1)

    # ## compute where the patch will start and end.
    start_x = int(np.max([x - np.floor(patch_size / 2), 0]))
    end_x = int(np.min([x + np.ceil(patch_size / 2), img_width]))
    start_y = int(np.max([y - np.floor(patch_size / 2), 0]))
    end_y = int(np.min([y + np.ceil(patch_size / 2), img_height]))

    mask = np.zeros((img_width, img_height, n_channels))
    mask[start_x:end_x, start_y:end_y, :] = 1

    # # create gaussian noise and apply it to the mask
    noise = scale * np.random.randn(*img_array.shape)
    mask = noise * mask

    # # apply the mask to the image
    img_array += mask
    img_array = np.clip(img_array, 0, 1)
    img_array *= 255.0

    img = PIL.Image.fromarray(np.uint8(img_array))

    return img
