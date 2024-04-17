import torch
#device = torch.device('cpu')
from mmdet.apis import init_detector
from mmengine.runner import Runner
from mmengine.config import Config, DictAction
import os.path as osp


config='projects/ViTDet/configs/vitdet_testing.py'
checkpoint = 'models/SSLhuge_satellite.pth'
#model = init_detector(config,checkpoint, device = 'cpu')
#trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#all_params = sum(p.numel() for p in model.parameters())
#print(f'Initializing model with {trainable_params}/{all_params} trainable params')
#print(model)
cfg = Config.fromfile(config)
#cfg.device='cpu'
cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(config))[0])
runner = Runner.from_cfg(cfg)
runner.train()