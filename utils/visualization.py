import os

import PIL
import torch
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

from configs.global_vars import IMG_MEAN
from utils.helpers import colorize_mask
from utils.palette import CityScpates_palette


class DeNormalize(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, tensor):
        IMG_MEAN = torch.from_numpy(self.mean.copy())
        IMG_MEAN, _ = torch.broadcast_tensors(IMG_MEAN.unsqueeze(1).unsqueeze(2), tensor)
        tensor = tensor+IMG_MEAN
        tensor = (tensor/255).float()
        tensor = torch.flip(tensor,(0,))
        return tensor

def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    mask = (img <= 0).squeeze()
    cm = plt.get_cmap(cmap)
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image

def subplotimg(ax, img, title, palette=CityScpates_palette, **kwargs):
    with torch.no_grad():
        if torch.is_tensor(img):
            img = img.cpu()
        if img.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])
            img = restore_transform(img)
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        else:
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette)

    ax.imshow(img, **kwargs)
    ax.set_title(title)

def save_image(folder, image, epoch, id, palette=CityScpates_palette):
    os.makedirs(folder, exist_ok=True)
    epoch_str = str(epoch) if epoch is not None else ""
    with torch.no_grad():
        if image.shape[0] == 3:
            restore_transform = transforms.Compose([
            DeNormalize(IMG_MEAN),
            transforms.ToPILImage()])
            image = restore_transform(image)
            image.save(os.path.join(folder, epoch_str + id + '.png'))
        elif image.shape[0] == 1:
            image = image.squeeze(0)
            if torch.is_tensor(image):
                image = image.numpy()
            # image = PIL.Image.fromarray(image * 255).convert("L")
            image = PIL.Image.fromarray((_colorize(image, cmap="plasma") * 255).astype(np.uint8))
            image.save(os.path.join(folder, epoch_str + id + '.png'))
        else:
            if torch.is_tensor(image):
                mask = image.numpy()
            else:
                mask = image
            colorized_mask = colorize_mask(mask, palette)
            colorized_mask.save(os.path.join(folder, epoch_str + id + '.png'))
