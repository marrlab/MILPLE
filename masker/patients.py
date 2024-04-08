dataset_type = 'ImageNet'
data_root = 'patients/'
img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

#img_size = 224
img_size = 144
estimation_pipeline = [
    #dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensorV2')
]

attribution_pipeline = [
    #dict(type='Resize', height=img_size, width=img_size, always_apply=True),
    dict(type='Normalize', always_apply=True, **img_norm_cfg),
    dict(type='ToTensorV2')
]

data = dict(
    data_loader=dict(batch_size=1, shuffle=True, num_workers=0),
    estimation=dict(
        type=dataset_type,
        img_root=data_root + 'estimation/',
        ind_to_cls_file=data_root + 'imagenet_class_index.json',
        pipeline=estimation_pipeline,
        with_bbox=False),
    attribution=dict(
        type=dataset_type,
        img_root=data_root + 'attribution/',
        annot_root=data_root + 'annotations/attribution/',
        ind_to_cls_file=data_root + 'imagenet_class_index.json',
        pipeline=attribution_pipeline,
        with_bbox=False))
