import argparse
import copy
import os
import os.path as osp
import sys
import time

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import mmcv
from mmcv.utils import Config, DictAction
from mmcv.runner import build_optimizer, load_checkpoint
from mmcv.parallel import MMDataParallel
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
# from mmseg.apis import single_gpu_test

from uda.models.generators import LightImgGenerator, FCDiscriminator
from uda.models.losses import StaticLoss, TVLoss, EXPLoss, SSIMLoss
from uda.datasets import ZurichPairDataset  # noqa

from evaluation import single_gpu_test


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter)**(power))


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs()**2)


def adjust_learning_rate(base_lr, optimizer, i_iter, max_iters, power):
    lr = lr_poly(base_lr, i_iter, max_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(base_lr, optimizer, i_iter, max_iters, power):
    lr = lr_poly(base_lr, i_iter, max_iters, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--load-from',
                        help='the checkpoint file to load weights from')
    parser.add_argument('--resume-from',
                        help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus',
                            type=int,
                            help='number of gpus to use '
                            '(only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids',
                            type=int,
                            nargs='+',
                            help='ids of gpus to use '
                            '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--options',
                        nargs='+',
                        action=DictAction,
                        help='custom options')
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(cfg.work_dir)

    # log_file
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    sys.stdout = Logger(log_file)

    model = build_segmentor(cfg.model.segmentor)
    checkpoint = load_checkpoint(model, cfg.checkpoint, map_location='cpu')

    model.train()
    model.to(device)

    lightnet = LightImgGenerator(**cfg.model.lightnet)
    lightnet.train()
    lightnet.to(device)

    model_D1 = FCDiscriminator(**cfg.model.discriminator)
    model_D1.train()
    model_D1.to(device)

    model_D2 = FCDiscriminator(**cfg.model.discriminator)
    model_D2.train()
    model_D2.to(device)

    cityscapes_dataset = build_dataset(cfg.data.cityscapes_train)
    dark_zurich_dataset = build_dataset(cfg.data.dark_zurich_pair)

    train_loader = build_dataloader(cityscapes_dataset,
                                    samples_per_gpu=cfg.data.samples_per_gpu,
                                    workers_per_gpu=cfg.data.workers_per_gpu,
                                    num_gpus=1,
                                    dist=False,
                                    shuffle=True,
                                    seed=cfg.data.seed)

    target_loader = DataLoader(dark_zurich_dataset,
                               batch_size=cfg.data.samples_per_gpu,
                               num_workers=cfg.data.workers_per_gpu,
                               persistent_workers=True,
                               shuffle=True,
                               pin_memory=True)

    # evaluation dataset
    if cfg.get('evaluation', None) is not None:
        val_dataset = build_dataset(cfg.data.test)
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=True)
        model.CLASSES = val_dataset.CLASSES
        model.PALETTE = val_dataset.PALETTE

    trainloader_iter = iter(train_loader)
    targetloader_iter = iter(target_loader)

    # segmentor_optimizer = build_optimizer(model, cfg.optimizer.segmentor)
    # d1_optimizer = build_optimizer(model_D1, cfg.optimizer.discriminator)
    # d2_optimizer = build_optimizer(model_D2, cfg.optimizer.discriminator)

    optimizer = optim.SGD(
        list(model.parameters()) + list(lightnet.parameters()),  # noqa
        lr=cfg.optimizer.segmentor.lr,
        momentum=cfg.optimizer.segmentor.momentum,
        weight_decay=cfg.optimizer.segmentor.weight_decay)

    d1_optimizer = optim.Adam(model_D1.parameters(),
                              lr=cfg.optimizer.discriminator.lr,
                              betas=cfg.optimizer.discriminator.betas)
    d2_optimizer = optim.Adam(model_D2.parameters(),
                              lr=cfg.optimizer.discriminator.lr,
                              betas=cfg.optimizer.discriminator.betas)
    optimizer.zero_grad()
    d1_optimizer.zero_grad()
    d2_optimizer.zero_grad()

    weights = torch.log(
        torch.FloatTensor([
            0.36869696, 0.06084986, 0.22824049, 0.00655399, 0.00877272,
            0.01227341, 0.00207795, 0.0055127, 0.15928651, 0.01157818,
            0.04018982, 0.01218957, 0.00135122, 0.06994545, 0.00267456,
            0.00235192, 0.00232904, 0.00098658, 0.00413907
        ])).cuda()
    weights = (torch.mean(weights) -
               weights) / torch.std(weights) * cfg.weights_cfg.std + 1.0

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255, weight=weights)

    static_loss = StaticLoss(num_classes=11, weights=weights[:11])

    loss_exp_z = EXPLoss(32)

    loss_TV = TVLoss()

    loss_SSIM = SSIMLoss()

    interp = nn.Upsample(size=cfg.crop_size,
                         mode='bilinear',
                         align_corners=True)
    interp_target = nn.Upsample(size=cfg.target_crop_size,
                                mode='bilinear',
                                align_corners=True)

    source_label = 0
    target_label = 1

    for i_iter in range(cfg.max_iters):

        start_time = time.time()

        loss_seg_value = 0
        loss_adv_target_value = 0
        loss_pseudo = 0
        loss_D_value1 = 0
        loss_D_value2 = 0

        optimizer.zero_grad()
        adjust_learning_rate(cfg.lr_config.segmentor_base_lr, optimizer,
                             i_iter, cfg.max_iters, cfg.lr_config.power)
        d1_optimizer.zero_grad()
        adjust_learning_rate_D(cfg.lr_config.discriminator_base_lr,
                               d1_optimizer, i_iter, cfg.max_iters,
                               cfg.lr_config.power)
        d2_optimizer.zero_grad()
        adjust_learning_rate_D(cfg.lr_config.discriminator_base_lr,
                               d2_optimizer, i_iter, cfg.max_iters,
                               cfg.lr_config.power)

        #  torch.autograd.set_detect_anomaly(True)  # 检查反向传播报错
        for sub_i in range(cfg.iter_size):
            # train G
            for param in model_D1.parameters():
                param.requires_grad = False
            for param in model_D2.parameters():
                param.requires_grad = False
            # train with target
            batch = next(targetloader_iter)

            images_day, img_day_metas, images_night, img_night_metas = batch[
                'img_day'], batch['img_day_metas'], batch['img_night'], batch[
                    'img_night_metas']

            images_day = images_day.to(device)

            # relight
            mean_light = images_night.mean()
            r = lightnet(images_day)
            enhanced_images_day = images_day + r
            loss_enhance = 10 * loss_TV(r) + torch.mean(
                loss_SSIM(enhanced_images_day, images_day)) + torch.mean(
                    loss_exp_z(enhanced_images_day, mean_light))

            pred_target_day = model.encode_decode(enhanced_images_day,
                                                  img_day_metas)

            pred_target_day = interp_target(pred_target_day)

            D_out_d = model_D1(F.softmax(pred_target_day, dim=1))
            D_label_d = torch.FloatTensor(
                D_out_d.data.size()).fill_(source_label).to(device)
            loss_adv_target_d = weightedMSE(D_out_d, D_label_d)

            loss = 0.01 * loss_adv_target_d + 0.01 * loss_enhance
            loss = loss / cfg.iter_size
            loss.backward()

            images_night = images_night.to(device)
            r = lightnet(images_night)
            enhanced_images_night = images_night + r
            loss_enhance = 10 * loss_TV(r) + torch.mean(
                loss_SSIM(enhanced_images_night, images_night)) + torch.mean(
                    loss_exp_z(enhanced_images_night, mean_light))

            pred_target_night = model.encode_decode(enhanced_images_night,
                                                    img_night_metas)
            pred_target_night = interp_target(pred_target_night)

            pseudo_prob = torch.zeros_like(pred_target_day)
            threshold = torch.ones_like(pred_target_day[:, :11, :, :]) * 0.2
            threshold[pred_target_day[:, :11, :, :] > 0.4] = 0.8
            pseudo_prob[:, :11, :, :] = threshold\
                * pred_target_day[:, :11, :, :].detach()\
                + (1 - threshold) * pred_target_night[:, :11, :, :].detach()
            pseudo_prob[:, 11:, :, :] = pred_target_night[:,
                                                          11:, :, :].detach()

            weights_prob = weights.expand(pseudo_prob.size()[0],
                                          pseudo_prob.size()[3],
                                          pseudo_prob.size()[2], 19)
            weights_prob = weights_prob.transpose(1, 3)
            pseudo_prob = pseudo_prob * weights_prob
            pseudo_gt = torch.argmax(pseudo_prob.detach(), dim=1)
            pseudo_gt[pseudo_gt >= 11] = 255

            D_out_n_19 = model_D2(F.softmax(pred_target_night, dim=1))
            D_label_n_19 = torch.FloatTensor(
                D_out_n_19.data.size()).fill_(source_label).to(device)
            loss_adv_target_n_19 = weightedMSE(
                D_out_n_19,
                D_label_n_19,
            )

            loss_pseudo = static_loss(pred_target_night[:, :11, :, :],
                                      pseudo_gt)
            loss = 0.01 * loss_adv_target_n_19\
                + loss_pseudo + 0.01 * loss_enhance
            loss = loss / cfg.iter_size
            loss.backward()
            loss_adv_target_value += loss_adv_target_n_19.item(
            ) / cfg.iter_size

            # train with source
            batch = next(trainloader_iter)

            images, labels, img_metas = batch['img'].data, batch[
                'gt_semantic_seg'].data, batch['img_metas'].data

            images = images[0].to(device)
            labels = labels[0].long().to(device).squeeze(1)
            img_metas = img_metas[0]

            relight_imgs = lightnet(images)
            enhanced_images = images + relight_imgs
            loss_enhance = 10 * loss_TV(relight_imgs) + torch.mean(
                loss_SSIM(enhanced_images, images)) + torch.mean(
                    loss_exp_z(enhanced_images, mean_light))

            seg_logits = model.encode_decode(enhanced_images, img_metas)

            pred_c = interp(seg_logits)

            loss_seg = seg_loss(pred_c, labels)

            loss = loss_seg + loss_enhance
            loss = loss / cfg.iter_size
            loss.backward()
            loss_seg_value += loss_seg.item() / cfg.iter_size

            # train D
            for param in model_D1.parameters():
                param.requires_grad = True
            for param in model_D2.parameters():
                param.requires_grad = True

            # train with source
            pred_c = pred_c.detach()
            D_out1 = model_D1(F.softmax(pred_c, dim=1))
            D_label1 = torch.FloatTensor(
                D_out1.data.size()).fill_(source_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / cfg.iter_size / 2
            loss_D1.backward()
            loss_D_value2 += loss_D1.item()

            pred_c = pred_c.detach()
            D_out2 = model_D2(F.softmax(pred_c, dim=1))
            D_label2 = torch.FloatTensor(
                D_out2.data.size()).fill_(source_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / cfg.iter_size / 2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

            # train with target
            pred_target_day = pred_target_day.detach()
            D_out1 = model_D1(F.softmax(pred_target_day, dim=1))
            D_label1 = torch.FloatTensor(
                D_out1.data.size()).fill_(target_label).to(device)
            loss_D1 = weightedMSE(D_out1, D_label1)
            loss_D1 = loss_D1 / cfg.iter_size / 2
            loss_D1.backward()
            loss_D_value1 += loss_D1.item()

            pred_target_night = pred_target_night.detach()
            D_out2 = model_D2(F.softmax(pred_target_night, dim=1))
            D_label2 = torch.FloatTensor(
                D_out2.data.size()).fill_(target_label).to(device)
            loss_D2 = weightedMSE(D_out2, D_label2)
            loss_D2 = loss_D2 / cfg.iter_size / 2
            loss_D2.backward()
            loss_D_value2 += loss_D2.item()

        optimizer.step()
        d1_optimizer.step()
        d2_optimizer.step()

        iter_time = time.time() - start_time
        eta = iter_time * (cfg.max_iters - i_iter)
        mins, s = divmod(eta, 60)
        hours, minute = divmod(mins, 60)
        days, hour = divmod(hours, 24)
        ETA = f'{int(days)}天{int(hour)}小时{int(minute)}分{int(s)}秒'

        print(
            'iter-[{0:8d}/{1:8d}], optimizer_lr={8:.8f}, optimizer_D1_lr={9:.8f}, optimizer_D1_lr={10:.8f}, loss_seg = {2:.3f}, loss_adv = {3:.3f}, loss_D1 = {4:.3f}, loss_D2 = {5:.3f}, loss_pseudo = {6:.3f}, ETA:{7}'  # noqa
            .format(i_iter, cfg.max_iters, loss_seg_value,
                    loss_adv_target_value, loss_D_value1, loss_D_value2,
                    loss_pseudo, ETA, optimizer.param_groups[0]['lr'],
                    d1_optimizer.param_groups[0]['lr'],
                    d2_optimizer.param_groups[0]['lr']))

        if cfg.get('evaluation', None):
            if i_iter % cfg.evaluation.iterval == 0:
                results = single_gpu_test(MMDataParallel(model,
                                                         device_ids=[0]),
                                          lightnet,
                                          val_dataloader,
                                          pre_eval=True,
                                          format_only=False)

                metric = val_dataset.evaluate(results, metric='mIoU')
                print(metric)
                model.train()

        if i_iter % cfg.checkpoint_config.iterval == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(),
                       os.path.join(cfg.work_dir, f'dannet_{str(i_iter)}.pth'))
            torch.save(
                lightnet.state_dict(),
                os.path.join(cfg.work_dir, f'dannet_light_{str(i_iter)}.pth'))
            torch.save(
                model_D1.state_dict(),
                os.path.join(cfg.work_dir, f'dannet_d1_{str(i_iter)}.pth'))
            torch.save(
                model_D2.state_dict(),
                os.path.join(cfg.work_dir, f'dannet_d2_{str(i_iter)}.pth'))


if __name__ == '__main__':
    main()
