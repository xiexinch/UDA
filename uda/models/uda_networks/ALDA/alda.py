import torch
from mmcv.runner import BaseModule
from mmseg.models import build_segmentor
from mmgen.models.common import set_requires_grad

from uda.models.uda_networks.ALDA.utils import calc_coeff

from ...builder import UDA_NETWORK, build_uda_loss
from .adversarial_network import MultiAdversarialNetwork, AdversarialNetwork


@UDA_NETWORK.register_module()
class ALDA(BaseModule):

    def __init__(self,
                 segmentor: dict,
                 discriminator: dict,
                 seg_loss: dict,
                 alda_loss: dict,
                 adv_weight=1.0,
                 multi_adv=True,
                 correct_loss=True,
                 reg_loss=True,
                 init_cfg=None):
        super().__init__()
        self.segmentor = build_segmentor(segmentor)
        if multi_adv:
            self.adversarial_model = MultiAdversarialNetwork(**discriminator)
        else:
            self.adversarial_model = AdversarialNetwork(**discriminator)
        self.alda_loss = build_uda_loss(alda_loss)

        self.correct_loss = correct_loss
        self.reg_loss = reg_loss
        self.adv_weight = adv_weight

    def train_step(self, source_batch, target_batch, curr_iter):
        source_imgs, source_img_metas, source_gt_seg_maps = source_batch
        target_imgs, target_img_metas, target_gt_seg_maps = target_batch

        source_features, source_seg_logits = self.segmentor(source_imgs,
                                                            source_img_metas,
                                                            return_loss=False)
        target_features, target_seg_logits = self.segmentor(target_imgs,
                                                            target_img_metas,
                                                            return_loss=False)

        features = torch.cat((source_features, target_features), dim=0)
        seg_logits = torch.cat((source_seg_logits, target_seg_logits), dim=0)

        ad_outs = self.discriminator(features)

        adv_loss, reg_loss, correct_loss = self.alda_loss(
            ad_outs, source_gt_seg_maps, seg_logits)

        if self.correct_loss:
            trade_off = calc_coeff(curr_iter, high=1)
            transfer_loss = self.adv_weight * (adv_loss +
                                               trade_off * correct_loss)
        else:
            transfer_loss = adv_loss

        if not self.reg_loss:
            set_requires_grad(self.segmentor, False)
            reg_loss.backward(retrain_graph=True)
            set_requires_grad(self.segmentor, True)

        source_ce_loss = self.seg_loss(source_seg_logits, source_gt_seg_maps)
        target_ce_loss = self.seg_loss(target_seg_logits, target_gt_seg_maps)
        ce_loss = torch.cat((source_ce_loss, target_ce_loss), dim=0)

        loss = transfer_loss + ce_loss
        loss.backward()
