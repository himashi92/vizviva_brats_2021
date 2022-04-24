import argparse
import os
import pathlib
import random

import SimpleITK as sitk
import numpy as np
import torch
import torch.optim
import torch.utils.data
from monai.transforms import (
    AsDiscrete
)
from torch.cuda.amp import autocast

import model
from dataset.batch_utils import pad_batch1_to_compatible_size
from dataset.brats import get_datasets_val
from model import get_norm_layer
from utils import reload_ckpt_bis, count_parameters

parser = argparse.ArgumentParser(description='Brats validation and testing dataset inference')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Unet', help='model architecture (default: Unet)')
parser.add_argument('--norm_layer', default='group')
parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
parser.add_argument('--width', default=48, help='base number of features for Unet (x2 per downsampling)', type=int)
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--on', default="val", choices=["val", "train", "test"])
parser.add_argument('--tta', action="store_true")
parser.add_argument('--seed', default=16111990)


def generate_segmentations(data_loader, model, args):
    metrics_list = []
    model = model.cuda()

    for i, batch in enumerate(data_loader):
        # measure data loading time
        inputs = batch["image"][0]
        patient_id = batch["patient_id"][0]
        ref_path = batch["seg_path"][0]
        crops_idx = batch["crop_indexes"]
        inputs, pads = pad_batch1_to_compatible_size(inputs)
        inputs = inputs.cuda()

        ref_seg_img = sitk.ReadImage(ref_path)
        ref_seg = sitk.GetArrayFromImage(ref_seg_img)

        with autocast():
            with torch.no_grad():
                pre_segs = model(inputs)
                pre_segs = torch.sigmoid(pre_segs)

        
        # remove pads
        maxz, maxy, maxx = pre_segs.size(2) - pads[0], pre_segs.size(3) - pads[1], pre_segs.size(4) - pads[2]
        pre_segs = pre_segs[:, :, 0:maxz, 0:maxy, 0:maxx].cpu()
        segs = torch.zeros((1, 3, ref_seg.shape[0], ref_seg.shape[1], ref_seg.shape[2]))
        segs[0, :, slice(*crops_idx[0]), slice(*crops_idx[1]), slice(*crops_idx[2])] = pre_segs[0]
        segs = segs[0].numpy() > 0.5

        et = segs[0]
        net = np.logical_and(segs[1], np.logical_not(et))
        ed = np.logical_and(segs[2], np.logical_not(segs[1]))
        labelmap = np.zeros(segs[0].shape)
        labelmap[et] = 4
        labelmap[net] = 1
        labelmap[ed] = 2
        labelmap = labelmap.astype(np.uint8)
        labelmap = sitk.GetImageFromArray(labelmap)

        ref_seg_img = sitk.ReadImage(ref_path)
        labelmap.CopyInformation(ref_seg_img)
        print(f"Writing {args.seg_folder}/{patient_id}.nii.gz")
        sitk.WriteImage(labelmap, f"{args.seg_folder}/{patient_id}.nii.gz")


def main(args):
    # setup
    random.seed(args.seed)
    ngpus = torch.cuda.device_count()
    if ngpus == 0:
        raise RuntimeWarning("This will not be able to run on CPU only")
    print(f"Working with {ngpus} GPUs")

    # Give the folder path of best model
    args.exp_name = "brats_2021"
    args.save_folder = pathlib.Path(f"./runs/{args.exp_name}/model_1")

    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)

    # Create model
    model_maker = getattr(model, args.arch)
    model_1 = model_maker(
        4, 3,
        width=args.width, norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)

    print(f"total number of trainable parameters {count_parameters(model_1)}")

    bench_dataset = get_datasets_val()
    bench_loader = torch.utils.data.DataLoader(bench_dataset, batch_size=1, num_workers=args.workers)

    print("Bench Test dataset number of batch:", len(bench_loader))

    reload_ckpt_bis(f'{str(args.save_folder)}/model_best.pth.tar', model_1)

    generate_segmentations(bench_loader, model_1, args)


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
