"""
Author: Sigve Rokenes
Date: February, 2019

Batch manager for pokemon generation

"""

import os, random
import skimage as sk
from skimage import io
from skimage import img_as_float
from skimage import transform


class PokeBatch:

    def __init__(self, resize=None):

        path = "data/processed/"
        paths = os.listdir(path)

        self.images = []
        self.next = 0

        for p in paths:
            img = sk.io.imread(path+p)
            if resize:
                img = transform.resize(img, resize)
            img = img_as_float(img)
            self.images.append(img)

    def shuffle(self):
        random.shuffle(self.images)

    def sample(self, size=1):
        return random.sample(self.images, size)

    def next_batch(self, batch_size):
        batch = self.images[self.next:self.next+batch_size]
        self.next += batch_size
        self.next %= len(self.images)
        return batch

    def num_examples(self):
        return len(self.images)