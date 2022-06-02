import os
import cv2
from data.transform import ColorHintTransform
import torch.utils.data as data

class ColorHintDataset(data.Dataset):
    def __init__(self, root_path, size, mode="train"):
        super(ColorHintDataset, self).__init__()

        self.root_path = root_path
        self.size = size
        self.mode = mode
        self.transforms = ColorHintTransform(self.size, self.mode)
        self.examples = None
        self.examples_step1 = None

        self.hint = None
        self.mask = None

        if self.mode == "train":
            train_dir = os.path.join(self.root_path, "train")
            self.examples = [os.path.join(self.root_path, "train", dirs) for dirs in os.listdir(train_dir)]

        elif self.mode == "val":
            val_dir = os.path.join(self.root_path, "val")
            self.examples = [os.path.join(self.root_path, "val", dirs) for dirs in os.listdir(val_dir)]

        elif self.mode == "test":
            # hint_dir = os.path.join(self.root_path, "test/hint")
            # mask_dir = os.path.join(self.root_path, "test/mask")
            hint_dir = os.path.join(self.root_path, "hint")
            mask_dir = os.path.join(self.root_path, "mask")
            self.hint = [os.path.join(self.root_path, "hint", dirs) for dirs in os.listdir(hint_dir)]
            self.mask = [os.path.join(self.root_path, "mask", dirs) for dirs in os.listdir(mask_dir)]
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode != "test":
            return len(self.examples)
        else:
            return len(self.hint)

    def __getitem__(self, idx):
        if self.mode == "test":
            hint_file_name = self.hint[idx]
            mask_file_name = self.mask[idx]
            hint_img = cv2.imread(hint_file_name)
            mask_img = cv2.imread(mask_file_name)

            input_l, input_hint, mask = self.transforms(hint_img, mask_img)

            sample = {"l": input_l, "hint": input_hint, "mask": mask,
                      "file_name": "image_%06d.png" % int(os.path.basename(hint_file_name).split('.')[0])}

        else:
            file_name = self.examples[idx]
            img = cv2.imread(file_name)
            l, ab, hint, mask = self.transforms(img)
            sample = {"l": l, "ab": ab, "hint": hint, "mask": mask, "filename":file_name}

        return sample