import warnings
import mmcv
import torch
from mmcv.runner import IterBasedRunner, RUNNERS, IterLoader, get_host_info


@RUNNERS.register_module()
class ALDAIterBasedRunner(IterBasedRunner):

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

    def run(self,
            source_data_loaders,
            target_data_loaders,
            workflow,
            max_iters=None,
            **kwargs):
        """Start running.
        """
        assert isinstance(source_data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)

        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')

        source_iter_loaders = [IterLoader(x) for x in source_data_loaders]
        target_iter_loaders = [IterLoader(x) for x in target_data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    if mode == 'train':
                        iter_runner(source_iter_loaders[i],
                                    target_iter_loaders[i], **kwargs)
                    elif mode == 'generate_pseudo_labels':
                        iter_runner(target_iter_loaders[i], )
                    elif mode == 'val':
                        iter_runner(target_iter_loaders[i], **kwargs)

    @torch.no_grad()
    def generate_pseudo_labels(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'generate_pseudo_labels'
        self.data_loader = data_loader
        data_batch = next(data_loader)
        outputs = self.model.inference_batch(data_batch, **kwargs)
        # TODO save pseudo labels
        self.outputs = outputs
        self.call_hook('after_generate_pseudo_labels')
        self._inner_iter += 1

    def train(self, source_data_loader, target_data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.source_loader = source_data_loader
        self.target_loader = target_data_loader
        source_batch = next(source_data_loader)
        target_batch = next(target_data_loader)
        self.call_hook('before_train_iter')
        outputs = self.model.train_step(source_batch, target_batch,
                                        self.optimizer, **kwargs)
        if not isinstance(outputs, dict):
            raise TypeError('model.train_step() must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
        self.call_hook('after_train_iter')
        self._inner_iter += 1
        self._iter += 1
