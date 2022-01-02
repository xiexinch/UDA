import copy
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.runner import HOOKS, Hook
from mmseg.datasets import build_dataset, build_dataloader


@HOOKS.register_module()
class PseudoLabelGenerationHook(Hook):

    def __init__(self,
                 target_dataset: dict,
                 pseudo_label_path: str,
                 threshold=0.9,
                 iter_num=4000):
        super().__init__()
        # generate pseudo labels per iter_num iters
        self.iter_num = iter_num
        self.dataset_cfg = copy.deepcopy(target_dataset)
        self.pseudo_label_path = pseudo_label_path
        self.threshold = threshold

    @torch.no_grad()
    def before_train_iter(self, runner):
        if runner.iter % self.iter_num == 0:
            dataset = build_dataset(self.dataset_cfg)
            data_loader = build_dataloader(dataset,
                                           samples_per_gpu=1,
                                           workers_per_gpu=2)
            model = runner.model.segmentor
            model.eval()
            labels = []
            max_pred_values = []
            prog_bar = mmcv.ProgressBar(len(dataset))
            mmcv.mkdir_or_exist(self.pseudo_label_path)
            loader_indices = data_loader.batch_sampler
            for batch_indices, data in zip(loader_indices, data_loader):

                with torch.no_grad():
                    features = model.encode_decode(**data)

                # value and label
                max_pred, seg_pred = torch.max(features, dim=1)

                # save result
                img_name = data['img_metas'][0].data[0][0]['filename'].replace(
                    '\\', '/').split('/')[-1].replace('_rgb_anon.png', '')
                label_path = osp.join(
                    self.pseudo_label_path,
                    f'{batch_indices[0]}_{img_name}_label.npy')
                max_pred_path = osp.join(
                    self.pseudo_label_path,
                    f'{batch_indices[0]}_{img_name}_conf.npy')
                labels.append(label_path)
                max_pred_values.append(max_pred_path)

                np.save(label_path, seg_pred.numpy())
                np.save(max_pred_path, max_pred.numpy())

                prog_bar.update()
