# dataset settings
cityscapes_type = 'CityscapesDataset'
cityscapes_data_root = 'data/cityscapes/'
zurich_dataset_type = 'DarkZurichDataset'
zurich_data_root = 'data/dark_zurich/'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
crop_size = (512, 1024)
target_crop_size = (540, 960)
# target_crop_size = (32, 32)
# crop_size = (32, 32)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
target_train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1080), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=target_crop_size, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=target_crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=target_crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img']),
]
source_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# mix dataset
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    seed=2022,
    cityscapes_train=dict(type='RepeatDataset',
                          times=500,
                          dataset=dict(type=cityscapes_type,
                                       data_root=cityscapes_data_root,
                                       img_dir='leftImg8bit/train',
                                       ann_dir='gtFine/train',
                                       pipeline=train_pipeline)),
    dark_zurich_pair=dict(
        type='ZurichPairDataset',
        data_root='data/dark_zurich/train/rgb_anon',
        pair_list_path='configs/_base_/datasets/zurich_dn_pair_train.csv',
        pipeline=target_train_pipeline,
        repeat_times=500),
    test=dict(type='DarkZurichDataset',
              data_root='data/dark_zurich/',
              img_dir='val/rgb_anon/val/night',
              ann_dir='val/gt/val/night',
              pipeline=test_pipeline))
