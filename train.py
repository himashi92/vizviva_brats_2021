import argparse
import os
import pathlib
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

import model
from dataset.batch_utils import determinist_collate
from dataset.brats import get_datasets
from learning_rate.poly_lr import poly_lr
from loss import EDiceLoss
from loss.adversarial_loss_gen import adv_loss_critic_v1
from loss.vat import vat_loss
from model import get_norm_layer
from model.critic import Discriminator
from utils import AverageMeter, ProgressMeter, save_checkpoint, reload_ckpt_bis, \
    count_parameters, save_metrics, save_args

parser = argparse.ArgumentParser(description='BRATS 2021 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='Unet', help='model architecture (default: Unet)')
parser.add_argument('--width', default=32, help='base number of features for Unet (x2 per downsampling)', type=int)
# DO not use data_aug argument this argument!!
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2).')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float, metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--lr_dis', type=float, default=5e-5, help='learning rate')
parser.add_argument('--wd', '--weight-decay', default=1e-03, type=float,
                    metavar='W', help='weight decay (default: 0)',
                    dest='weight_decay')
# Warning: untested option!!
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint. Warning: untested option')
parser.add_argument('--devices', default='0', type=str, help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument('--seed', default=16111990, help="seed for train/val split")
parser.add_argument('--warmup', default=5, type=int, metavar='N', help='number of warmup epochs')
parser.add_argument('--disable-cos', action='store_true', help='disable cosine lr schedule')
parser.add_argument('--warm', default=3, type=int, help="number of warming up epochs")
parser.add_argument('--val', default=1, type=int, help="how often to perform validation step")
parser.add_argument('--fold', default=0, type=int, help="Split number (0 to 4)")
parser.add_argument('--norm_layer', default='inorm')
parser.add_argument('--optim', choices=['adam', 'sgd', 'adamw'], default='adam')
parser.add_argument('--com', help="add a comment to this run!")
parser.add_argument('--dropout', type=float, help="amount of dropout to use", default=0.)
parser.add_argument('--warm_restart', action='store_true', help='use scheduler warm restarts with period of 30')
parser.add_argument('--full', action='store_true', help='Fit the network on the full training set')
parser.add_argument('--lambda_adv', type=float, default=0.3, help='scalar constant adversarial loss')
parser.add_argument('--lambda_vat', type=float, default=0.2, help='scalar constant vat loss')


def main(args):
    # setup
    ngpus = torch.cuda.device_count()
    print(f"Working with {ngpus} GPUs")

    args.exp_name = "brats_2021".format(args.lambda_adv, args.lambda_vat)
    args.save_folder = pathlib.Path(f"./runs/{args.exp_name}/model_1")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    args.seg_folder = args.save_folder / "segs"
    args.seg_folder.mkdir(parents=True, exist_ok=True)
    args.save_folder = args.save_folder.resolve()
    save_args(args)
    t_writer_1 = SummaryWriter(str(args.save_folder))

    # Create model
    print(f"Creating {args.arch}")

    model_maker = getattr(model, args.arch)
    model_1 = model_maker(
        4, 3,
        width=args.width, norm_layer=get_norm_layer(args.norm_layer), dropout=args.dropout)

    print(f"total number of trainable parameters {count_parameters(model_1)}")
    print(f"scalar constant agreement loss {args.lambda_vat}")
    print(f"scalar constant adversarial loss {args.lambda_adv}")

    model_1 = model_1.cuda()

    model_file = args.save_folder / "model.txt"
    with model_file.open("w") as f:
        print(model_1, file=f)

    criterion = EDiceLoss().cuda()
    metric = criterion.metric
    print(metric)
    params = model_1.parameters()

    if args.optim == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=0)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.99, nesterov=True)
    elif args.optim == "adamw":
        print(f"weight decay argument will not be used. Default is 11e-2")
        optimizer = torch.optim.AdamW(params, lr=args.lr)

    critic = Discriminator()
    critic = critic.cuda()
    dis_optimizer = torch.optim.RMSprop(critic.parameters(), args.lr_dis)

    full_train_dataset, l_val_dataset, bench_dataset = get_datasets(args.seed,fold_number=args.fold)
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(l_val_dataset, batch_size=1, shuffle=False,
                                             pin_memory=True, num_workers=args.workers, collate_fn=determinist_collate)

    print("Val dataset number of batch:", len(val_loader))
    print("Full Labeled Train dataset number of batch:", len(train_loader))

    # create grad scaler
    scaler = GradScaler()

    # Actual Train loop
    best_1 = np.inf
    patients_perf = []

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    print("start training now!")

    for epoch in range(args.epochs):
        try:
            # do_epoch for one epoch
            ts = time.perf_counter()

            # Setup
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_ = AverageMeter('Loss', ':.4e')

            mode = "train" if model_1.training else "val"
            batch_per_epoch = len(train_loader)
            progress = ProgressMeter(
                batch_per_epoch,
                [batch_time, data_time, losses_],
                prefix=f"{mode} Epoch: [{epoch}]")

            end = time.perf_counter()
            metrics = []

            optimizer.param_groups[0]['lr'] = poly_lr(epoch, args.epochs, args.lr, 0.9)

            for i, batch in enumerate(zip(train_loader)):
                torch.cuda.empty_cache()
                # measure data loading time
                data_time.update(time.perf_counter() - end)

                inputs_S1, labels_S1 = batch[0]["image"].float(), batch[0]["label"].float()

                inputs_S1, labels_S1 = Variable(inputs_S1), Variable(labels_S1)
                inputs_S1, labels_S1 = inputs_S1.cuda(), labels_S1.cuda()

                optimizer.zero_grad()

                segs_S1 = model_1(inputs_S1)

                loss_sup = criterion(segs_S1, labels_S1)
                loss_vat = vat_loss(model_1, inputs_S1, labels_S1, eps=2.5).cuda()

                critic_segs_1 = torch.nn.functional.interpolate(torch.sigmoid(critic(segs_S1)),
                                                                (segs_S1.shape[2], segs_S1.shape[3], segs_S1.shape[4]),
                                                                mode='trilinear', align_corners=False)

                critic_segs_3 = torch.nn.functional.interpolate(torch.sigmoid(critic(labels_S1)),
                                                           (labels_S1.shape[2], labels_S1.shape[3], labels_S1.shape[4]),
                                                           mode='trilinear', align_corners=False)

                adversarial_loss = 0.5 * adv_loss_critic_v1(critic_segs_1, critic_segs_3, labels_S1)

                loss_ = loss_sup + args.lambda_vat * abs(loss_vat) + args.lambda_adv * adversarial_loss

                t_writer_1.add_scalar(f"Loss/{mode}{''}",
                                      loss_.item(),
                                      global_step=batch_per_epoch * epoch + i)

                # measure accuracy and record loss_
                if not np.isnan(loss_.item()):
                    losses_.update(loss_.item())
                else:
                    print("NaN in model loss!!")

                # compute gradient and do SGD step
                scaler.scale(loss_).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                t_writer_1.add_scalar("lr", optimizer.param_groups[0]['lr'], global_step=epoch * batch_per_epoch + i)

                del critic_segs_1, critic_segs_3, loss_
                gc.collect()

                dis_optimizer.zero_grad()

                adversarial_loss = adversarial_loss.detach()
                adversarial_loss = Variable(adversarial_loss, requires_grad=True)

                adversarial_loss.backward()
                dis_optimizer.step()

                del segs_S1, labels_S1, adversarial_loss
                gc.collect()

                # measure elapsed time
                batch_time.update(time.perf_counter() - end)
                end = time.perf_counter()
                # Display progress
                progress.display(i)

            t_writer_1.add_scalar(f"SummaryLoss/train", losses_.avg, epoch)

            te = time.perf_counter()
            print(f"Train Epoch done in {te - ts} s")
            torch.cuda.empty_cache()

            # Validate at the end of epoch every val step
            if (epoch + 1) % args.val == 0:
                validation_loss_1 = step(val_loader, model_1, criterion, metric, epoch, t_writer_1,
                                             save_folder=args.save_folder,
                                             patients_perf=patients_perf)

                if scheduler is not None:
                    scheduler.step(validation_loss_1)

                t_writer_1.add_scalar(f"SummaryLoss", validation_loss_1, epoch)

                if validation_loss_1 < best_1:
                    best_1 = validation_loss_1
                    model_dict = model_1.state_dict()
                    save_checkpoint(
                        dict(
                            epoch=epoch, arch=args.arch,
                            state_dict=model_dict,
                            optimizer=optimizer.state_dict(),
                            scheduler=scheduler.state_dict(),
                        ),
                        save_folder=args.save_folder, )

                ts = time.perf_counter()
                print(f"Val epoch done in {ts - te} s")
                torch.cuda.empty_cache()

        except KeyboardInterrupt:
            print("Stopping training loop, doing benchmark")
            break

    try:
        df_individual_perf = pd.DataFrame.from_records(patients_perf)
        print(df_individual_perf)
        df_individual_perf.to_csv(f'{str(args.save_folder)}/patients_indiv_perf.csv')
        reload_ckpt_bis(f'{str(args.save_folder)}/model_best.pth.tar', model_1)
        torch.cuda.empty_cache()
    except KeyboardInterrupt:
        print("Stopping right now!")


def step(data_loader, model, criterion: EDiceLoss, metric, epoch, writer, scaler=None,
         scheduler=None, save_folder=None, patients_perf=None):
    # Setup
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    mode = "val"
    batch_per_epoch = len(data_loader)
    progress = ProgressMeter(
        batch_per_epoch,
        [batch_time, data_time, losses],
        prefix=f"{mode} Epoch: [{epoch}]")

    end = time.perf_counter()
    metrics = []

    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.perf_counter() - end)

        targets = batch["label"].float()
        targets = targets.cuda()
        inputs = batch["image"].float()
        patient_id = batch["patient_id"]

        inputs = inputs.cuda()
        model.eval()
        with torch.no_grad():
            segs = model(inputs)
            loss_ = criterion(segs, targets)

        if patients_perf is not None:
            patients_perf.append(
                dict(id=patient_id[0], epoch=epoch, split=mode, loss=loss_.item())
            )

        writer.add_scalar(f"Loss/{mode}{''}",
                          loss_.item(),
                          global_step=batch_per_epoch * epoch + i)

        # measure accuracy and record loss_
        if not np.isnan(loss_.item()):
            losses.update(loss_.item())
        else:
            print("NaN in model loss!!")

        metric_ = metric(segs, targets)
        metrics.extend(metric_)

        # measure elapsed time
        batch_time.update(time.perf_counter() - end)
        end = time.perf_counter()
        # Display progress
        progress.display(i)

    save_metrics(epoch, metrics, writer, epoch, False, save_folder)
    writer.add_scalar(f"SummaryLoss/val", losses.avg, epoch)

    return losses.avg


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
