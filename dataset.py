import cv2
import numpy as np
import os

class Dataset:
    def __init__(self, image_dir, image_size = 256):
        self.image_dir = image_dir
        self.image_names = os.listdir(image_dir)
        self.no_images = len(self.image_names)
        self.image_size = 0
        self.image_size = image_size

    def get_batch(self, batch_size = 4, index = None):
        if index == None:
            index = np.random.choice(self.no_images, batch_size, False)
        else:
            if index + batch_size <= self.no_images:
                index = np.arange(batch_size) + index
            else:
                index = np.arange(self.no_images - index) + index
        return self.get_image_from_index(index)

    def get_image_from_index(self, index):
        grays = np.zeros(shape = [len(index), 256, 256, 1], dtype = np.float32)
        colors = np.zeros(shape = [len(index), 256, 256, 3], dtype = np.float32)

        for i, idx in enumerate(index):
            image_path = os.path.join(self.image_dir, self.image_names[idx])
            # Loading
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            # Reize to (image_size, image_size)
            image = cv2.resize(image, (self.image_size, self.image_size))
            gray = cv2.resize(gray, (self.image_size, self.image_size))
            # Scale to [-1..1]
            image = (image.astype(np.float32) - 127.5) / 127.5
            gray = (gray.astype(np.float32) - 127.5) / 127.5
            # Assign to output
            colors[i] = image
            grays[i, :, :,0] = gray

        return colors, grays
