import argparse
import json
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
from collections import OrderedDict
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo

from configs.global_vars import IMG_MEAN
from model import get_model
from data import get_data_path, get_loader
import torchvision.transforms as transform

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d
from utils.visualization import save_image

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UDA evaluation script")
    parser.add_argument("-m", "--model", type=str, default="deeplabv2_synthia",
                        choices=["deeplabv2_gta", "deeplabv2_synthia"],
                        help="Model to evaluate")
    parser.add_argument("-p", "--model-path", type=str, default=None, required=True,
                        help="Model checkpoint to evaluate")
    parser.add_argument("-c", "--override-config", type=str, default=None,
                        help="Override model config")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-predictions", action="store_true",
                        help="save output images")
    parser.add_argument("--full-resolution", action="store_true",
                        help="evaluate at full resolution")
    return parser.parse_args()

class VOCColorize(object):
    def __init__(self, n=22):
        self.cmap = color_map(22)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def get_label_vector(target, nclass):
    # target is a 3D Variable BxHxW, output is 2D BxnClass
    hist, _ = np.histogram(target, bins=nclass, range=(0, nclass-1))
    vect = hist>0
    vect_out = np.zeros((21,1))
    for i in range(len(vect)):
        if vect[i] == True:
            vect_out[i] = 1
        else:
            vect_out[i] = 0

    return vect_out

def get_iou(data_list, class_num, dataset, save_path=None):
    from multiprocessing import Pool
    from utils.metric import ConfusionMatrix

    ConfM = ConfusionMatrix(class_num)
    f = ConfM.generateM
    pool = Pool()
    m_list = pool.map(f, data_list)
    pool.close()
    pool.join()

    for m in m_list:
        ConfM.addM(m)

    aveJ, j_list, M = ConfM.jaccard()

    classes = np.array(("road", "sidewalk",
        "building", "wall", "fence", "pole",
        "traffic_light", "traffic_sign", "vegetation",
        "terrain", "sky", "person", "rider",
        "car", "truck", "bus",
        "train", "motorcycle", "bicycle"))


    for i, iou in enumerate(j_list):
        print('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]))

    print('meanIOU: ' + str(aveJ) + '\n')
    if save_path:
        with open(save_path, 'w') as f:
            for i, iou in enumerate(j_list):
                f.write('class {:2d} {:12} IU {:.2f}'.format(i, classes[i], 100*j_list[i]) + '\n')
            f.write('meanIOU: ' + str(aveJ) + '\n')
    return aveJ, j_list

def evaluate(model, dataset, config, save_dir=None):
    if "evaluation" not in config:
        config["evaluation"] = {}
    label_size = tuple(config["evaluation"].get("label_size", (512, 1024)))
    batch_size = config["evaluation"].get("batch_size", 1)
    save_n_predictions = config["evaluation"].get("save_n_predictions",0)

    if dataset == 'cityscapes':
        num_classes = 19
        assert label_size in [(1024, 2048), (512, 1024)]
        data_loader = get_loader('cityscapes')
        data_path = get_data_path('cityscapes')
        test_dataset = data_loader(data_path, img_size=(512, 1024), label_size=label_size, img_mean = IMG_MEAN,
                                   is_transform=True, split='val')
        testloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        interp = nn.Upsample(size=label_size, mode='bilinear', align_corners=True)
        ignore_label = 250

    elif dataset == 'gta':
        num_classes = 19
        assert label_size == (720, 1280)
        data_loader = get_loader('gta')
        data_path = get_data_path('gta')
        test_dataset = data_loader(data_path, list_path = './data/gta5_list/train.txt', img_size=(1280,720), mean=IMG_MEAN)
        testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)
        interp = nn.Upsample(size=(720,1280), mode='bilinear', align_corners=True)
        ignore_label = 255

    print('Evaluating, found ' + str(len(testloader)) + ' batches.')

    data_list = []
    colorize = VOCColorize()

    total_loss = []

    n_saved = 0
    if save_n_predictions != 0:
        os.makedirs(os.path.join(save_dir, 'predictions'), exist_ok=True)
    for index, batch in enumerate(testloader):
        images, labels, size, name, _ = batch
        size = size[0]
        with torch.no_grad():
            outputs  = model(Variable(images).cuda())
            if isinstance(outputs, dict):
                outputs = interp(outputs["S"])
            else:
                outputs = interp(outputs)

            labels_cuda = Variable(labels.long()).cuda()
            criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  
            for image, output, label, label_cuda in zip(images, outputs, labels, labels_cuda):
                loss = criterion(output.unsqueeze(0), label_cuda.unsqueeze(0))
                total_loss.append(loss.item())

                output = output.cpu().data.numpy()
                gt = np.asarray(label.numpy(), dtype=np.int)

                output = output.transpose(1,2,0)
                output = np.asarray(np.argmax(output, axis=2), dtype=np.int)

                if n_saved < save_n_predictions or save_n_predictions == -1:
                    out_dir = os.path.join(save_dir, "predictions")
                    save_image(out_dir, image, None, f'{n_saved}_0img')
                    save_image(out_dir, output, None, f'{n_saved}_1pred')
                    save_image(out_dir, label, None, f'{n_saved}_2gt')
                    n_saved += 1

                data_list.append([gt.flatten(), output.flatten()])

        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

    if save_dir:
        filename = os.path.join(save_dir, 'result.txt')
    else:
        filename = None
    mIoU, cIoU = get_iou(data_list, num_classes, dataset, filename)
    loss = np.mean(total_loss)
    return mIoU, cIoU, loss

def main():
    """Create the model and start the evaluation process."""

    gpu0 = args.gpu

    os.makedirs(save_dir, exist_ok=True)

    #model = torch.nn.DataParallel(Res_Deeplab(num_classes=num_classes), device_ids=args.gpu)
    model = get_model(args.model)(num_classes=num_classes)

    checkpoint = torch.load(args.model_path)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['model'])

    model.cuda()
    model.eval()

    evaluate(model, dataset, config=config, save_dir=save_dir)


if __name__ == '__main__':
    args = get_arguments()

    if args.override_config is not None:
        config = json.load(open(args.override_config))
    else:
        config = torch.load(args.model_path)['config']
    if "evaluation" not in config:
        config["evaluation"] = {}
    if args.full_resolution:
        config["evaluation"]["label_size"] = {"cityscapes": [1024, 2048], "gta": [1052, 1914]}[config["dataset"]]
    if args.save_predictions:
        config["evaluation"]["save_n_predictions"] = -1

    dataset = config['dataset']
    num_classes = 19

    ignore_label = config['ignore_label']
    save_dir = args.model_path.rsplit('/', 1)[0]
    print(save_dir)

    main()
