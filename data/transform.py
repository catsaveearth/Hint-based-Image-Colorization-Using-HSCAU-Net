import cv2
import random
import numpy as np
from torchvision import transforms


class ColorHintTransform(object):
    def __init__(self, size=256, mode="train"):
        super(ColorHintTransform, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]  # 0, 1~2
        return l, ab

    def hint_mask(self, bgr, threshold=[0.95, 0.97, 0.99]): 
        h, w, c = bgr.shape
        mask_threshold = random.choice(threshold)
        mask = np.random.random([h, w, 1]) > mask_threshold
        return mask

    def img_to_mask(self, mask_img):
        mask = mask_img[:, :, 0, np.newaxis] >= 255
        return mask

    def __call__(self, img, mask_img=None):
        threshold = [0.95, 0.97, 0.99] #
        if (self.mode == "train") | (self.mode == "val"):

            if self.mode == "train":
                threshold = [0.97, 0.99, 0.995]
            else:
                threshold = [0.995]

            image = cv2.resize(img, (self.size, self.size))
            mask = self.hint_mask(image, threshold)

            hint_image = image * mask

            l, ab = self.bgr_to_lab(image)
            l_hint, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab), self.transform(ab_hint), self.transform(mask) #transform => to_tensor

        elif self.mode == "test":
            image = cv2.resize(img, (self.size, self.size))
            mask = self.img_to_mask(mask_img)
            
            hint_image = image * mask

            l, _ = self.bgr_to_lab(image) 
            _, ab_hint = self.bgr_to_lab(hint_image)

            return self.transform(l), self.transform(ab_hint), self.transform(mask)

        else:
            return NotImplementedError
