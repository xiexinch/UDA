_base_ = [
    '../_base_/methods/deeplabv3_r50-d8.py',
    '../_base_/datasets/city_zurich_pair.py', '../default_runtime.py'
]

weights_cfg = dict(std=0.05)

optimizer = dict(segmentor=dict(type='SGD',
                                lr=2.5e-4,
                                momentum=0.9,
                                weight_decay=0.0005),
                 discriminator=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999)))
# lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

lr_config = dict(segmentor_base_lr=2.5e-4,
                 discriminator_base_lr=1e-4,
                 power=0.9)
max_iters = 20000
iter_size = 1
checkpoint_config = dict(iterval=2000)
evaluation = dict(iterval=200)
checkpoint = 'checkpoints/deeplabv3_r50-d8_512x1024_80k_cityscapes_20200606_113404-b92cfdd4.pth'
