import os
import torch
import cv2
import numpy as np
import scipy.io as io
import utils.scipymisc as m

from torch.utils import data

from data.city_utils import recursive_glob
from data.augmentations import *

class cityscapesLoader(data.Dataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    class_names = [
        "unlabelled",
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic_light",
        "traffic_sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {"cityscapes": [73.15835921, 82.90891754, 72.39239876],}

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(512, 1024),
        label_size=None,
        img_norm=False,
        augmentations=None,
        return_id=False,
        disparity=False,
        img_mean = np.array([73.15835921, 82.90891754, 72.39239876])
    ):
        """__init__

        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        assert self.img_size[0] <= self.img_size[1]
        self.label_size = self.img_size if label_size is None else label_size
        self.mean = img_mean
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit_trainvaltest","leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine_trainvaltest", "gtFine", self.split
        )
        self.depth_base =  os.path.join(
            self.root, "depth", self.split
        )
        self.disparity_base =  os.path.join(
            self.root, "disparity", self.split
        )

        with open(f'./data/cityscapes_list/{split}.txt') as fh:
            self.files[split] = [os.path.join(self.images_base, f) for f in fh.read().splitlines()]

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception(
                "No files for split=[%s] found in %s" % (split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files[split]), split))

        self.return_id = return_id
        self.disparity = disparity

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2], # temporary for cross validation
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = m.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = m.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(lbl)
        
        # Placeholder waiting for data
        if self.split=="train" and not self.disparity:
            ## Stereo Depth
            depth_path = os.path.join(
                self.depth_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path)[:-15] + "depth_stereoscopic.mat",
            )  
            depth = io.loadmat(depth_path)["depth_map"]  
            depth = np.clip(depth, 0., 655.35)
            depth[depth<0.1] = 655.35
            depth = 655.36 / (depth + 0.01)
        elif self.split=="train" and self.disparity:
            # Monocular depth: in disparity form 0 - 65535
            depth_path = os.path.join(
                self.disparity_base,
                img_path.split(os.sep)[-2],
                os.path.basename(img_path),
            ) 
            depth = cv2.imread(depth_path, flags=cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256. + 1.
            if depth.shape != lbl.shape:
                depth = cv2.resize(depth, lbl.shape[::-1], interpolation=cv2.INTER_NEAREST)
        else:
            # Dummpy depth not used
            depth = 0. * np.array(lbl.copy(), dtype=np.float32)

        if self.augmentations is not None:
            img, lbl, depth = self.augmentations(img, lbl, depth)

        if self.is_transform:
            img, lbl, depth = self.transform(img, lbl, depth)

        img_name = img_path.split('/')[-1]
        if self.return_id:
            return img, lbl, img_name, img_name, index
        
        if "val" not in self.split:
            return img, lbl, img_path, lbl_path, depth
        else:
            return img, lbl, img_path, lbl_path, img_name

    def transform(self, img, lbl, depth):
        """transform

        :param img:
        :param lbl:
        """
        img = m.imresize(
            img, (self.img_size[0], self.img_size[1])
        )  # uint8 with RGB mode
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        assert img.shape[0] <= img.shape[1]
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.label_size[0], self.label_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)
        depth = m.imresize(depth, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        depth = depth.astype(float)
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        depth = torch.from_numpy(depth).float()

        return img, lbl, depth

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

'''
if __name__ == "__main__":
    import torchvision
    import matplotlib.pyplot as plt

    augmentations = Compose([Scale(2048), RandomRotate(10), RandomHorizontallyFlip()])

    local_path = "./data/city_dataset/"
    dst = cityscapesLoader(local_path, is_transform=True, augmentations=augmentations)
    bs = 4
    trainloader = data.DataLoader(dst, batch_size=bs, num_workers=0)
    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs = imgs.numpy()[:, ::-1, :, :]
        imgs = np.transpose(imgs, [0, 2, 3, 1])
        f, axarr = plt.subplots(bs, 2)
        for j in range(bs):
            axarr[j][0].imshow(imgs[j])
            axarr[j][1].imshow(dst.decode_segmap(labels.numpy()[j]))
        plt.show()
        a = raw_input()
        if a == "ex":
            break
        else:
            plt.close()
'''
