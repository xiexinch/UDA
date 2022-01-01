from mmseg.models import SEGMENTORS, EncoderDecoder


@SEGMENTORS
class UDAEncoderDecoder(EncoderDecoder):

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(backbone,
                         decode_head,
                         neck=neck,
                         auxiliary_head=auxiliary_head,
                         train_cfg=train_cfg,
                         test_cfg=test_cfg,
                         pretrained=pretrained,
                         init_cfg=init_cfg)
