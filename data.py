"""
Author: Sigve Rokenes
Date: February, 2019

Data preparation from pokemon generation
This version uses flipping only for data augmentation.

"""

import os
import skimage as sk
from skimage import exposure
from skimage import io


def split_sheet(image, size):
    sprites = []
    w, h = image.shape[0], image.shape[1]
    for x in range(int(w / size)):
        for y in range(int(h / size)):
            part = image[x * size:x * size + size, y * size:y * size + size]
            sprites.append(part)
    return sprites


if __name__ == "__main__":

    print("Generating data...")

    raw_image_path = "data/raw/"
    processed_image_path = "data/processed/"

    image_data = {
        "sheet_01.png": 56,
        "sheet_02.png": 80,
        "sheet_03.png": 64
    }

    image_number = 0

    for f in image_data:
        img = sk.io.imread(os.path.join(raw_image_path, f))
        spr = split_sheet(img, image_data[f])

        print("Extracted", len(spr), "sprites from sheet", f)

        for n in range(len(spr)):
            if not sk.exposure.is_low_contrast(spr[n]):
                flip = spr[n][:, ::-1]
                sk.io.imsave(processed_image_path + "{:6d}.png".format(image_number), spr[n])
                sk.io.imsave(processed_image_path + "{:6d}.png".format(image_number+1), flip)
                image_number += 2

    print("Data generation complete,", image_number, "images generated.")
