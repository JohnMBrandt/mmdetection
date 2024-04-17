_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    #'./tree-verification.py',
    './balloon-verification.py'
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=False)
norm_cfg = dict(type='LN2d', requires_grad=True)
image_size = (256, 256)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone=dict(
        type='SSLVisionTransformer',
        img_size=256,
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        #drop_path_rate=0.1,
        #window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        pretrained = None,
        #pretrained = '../HighResCanopyHeight-main/saved_checkpoints/SSLhuge_satellite.pth',
        #norm_cfg=backbone_norm_cfg,
        out_indices=[
            9, 16, 22, 29,
        ],
        init_cfg=dict(
            type='Pretrained', checkpoint='../HighResCanopyHeight-main/saved_checkpoints/SSLhuge_satellite.pth')),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=1280,
        in_channels=[320, 640, 1280, 1280],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

#custom_hooks = [dict(type='Fp16CompresssionHook')]