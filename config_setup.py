from mmengine import Config
from mmengine.runner import set_random_seed

# load the config and path file that matches to the type of model you chose
cfg = Config.fromfile('./configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py')
cfg.load_from = './checkpoints/rtmdet-ins_l_8xb32-300e_coco_20221124_103237-78d1d652.pth'

# designate your working folder to save the trained model weights
cfg.work_dir = './work_dir'

# hyperparameters that you need to customize based on your own cases
cfg.max_epochs = 150
cfg.stage2_num_epochs = 7
cfg.train_dataloader.batch_size = 4
cfg.train_dataloader.num_workers = 2

scale_factor = cfg.train_dataloader.batch_size / (8 * 32)

cfg.base_lr *= scale_factor
cfg.optim_wrapper.optimizer.lr = cfg.base_lr

# rtmdet model has 4 backbone stages, set ths=is to 4 to freeze all backbone stages and 0 to train from scratch
cfg.model.backbone.frozen_stages = 4
# adjust the number of classes based on your own dataset
cfg.model.bbox_head.num_classes = 4

# BN - training on a single GPU  SYNCBN - training on multiple GPUs
cfg.norm_cfg = dict(type='BN', requires_grad=True)

'''
Backboard
Basketball
Hoop
Person
'''

# trailing comma is necessary
# palette colors in rgb values, pay attention not to have duplicate colors
# the class names must be arranged in the same order as listed in corresponding dataset
cfg.metainfo = {
    'classes': ('Backboard', 'Basketball', 'Hoop', 'Person',),
    'palette': [
        (235, 137, 52),(229, 235, 52),(226, 52, 235),(52, 137, 235),    
    ]
}

# link the data root to your dataset
cfg.data_root = './data'

cfg.train_dataloader.dataset.ann_file = 'basketball/train/_annotations.coco.json'
cfg.train_dataloader.dataset.data_root = cfg.data_root
cfg.train_dataloader.dataset.data_prefix.img = 'basketball/train/'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader.dataset.ann_file = 'basketball/test/_annotations.coco.json'
cfg.test_dataloader.dataset.data_root = cfg.data_root
cfg.test_dataloader.dataset.data_prefix.img = 'basketball/test/'
cfg.test_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'basketball/valid/_annotations.coco.json'
cfg.val_dataloader.dataset.data_root = cfg.data_root
cfg.val_dataloader.dataset.data_prefix.img = 'basketball/valid/'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_evaluator.ann_file = cfg.data_root+'/'+'basketball/test/_annotations.coco.json'
cfg.test_evaluator.metric = ['segm']

cfg.val_evaluator.ann_file = cfg.data_root+'/'+'basketball/valid/_annotations.coco.json'
cfg.val_evaluator.metric = ['segm']

cfg.default_hooks.checkpoint = dict(type='CheckpointHook', interval=10, max_keep_ckpts=2, save_best='auto')
cfg.default_hooks.logger.interval = 20

cfg.custom_hooks[1].switch_epoch = 300 - cfg.stage2_num_epochs

cfg.train_cfg.max_epochs = cfg.max_epochs
cfg.train_cfg.val_begin = 20
cfg.train_cfg.val_interval = 2
cfg.train_cfg.dynamic_intervals = [(300 - cfg.stage2_num_epochs, 1)]

# cfg.train_dataloader.dataset = dict(dict(type='RepeatDataset',times=5,dataset = cfg.train_dataloader.dataset))

cfg.param_scheduler[0].end = 100

cfg.param_scheduler[1].eta_min = cfg.base_lr * 0.05
cfg.param_scheduler[1].begin = cfg.max_epochs // 2
cfg.param_scheduler[1].end = cfg.max_epochs
cfg.param_scheduler[1].T_max = cfg.max_epochs //2

set_random_seed(0, deterministic=False)

# if you wan

#cfg.visualizer.vis_backends.append({"type":'WandbVisBackend'})
'''
add your own visualization backend
'''
#cfg.visualizer.vis_backends.pop()
#cfg.visualizer.vis_backends.append({"type":'WandbVisBackend'})

#------------------------------------------------------
#output your config file named following the rules introduced in readme
config=f'./configs/rtmdet/rtmdet-ins_l_1xb4-150e_LVBB.py'
with open(config, 'w') as f:
    f.write(cfg.pretty_text)