import torch
from mmcv.runner import IterBasedRunner, RUNNERS, IterLoader


@RUNNERS.register_module()
class UDAIterBasedRunner(IterBasedRunner):

    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super().__init__(model,
                         batch_processor=batch_processor,
                         optimizer=optimizer,
                         work_dir=work_dir,
                         logger=logger,
                         meta=meta,
                         max_iters=max_iters,
                         max_epochs=max_epochs)
