import mmcv
import mmengine
import argparse
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

def parse_args():
    parser = argparse.ArgumentParser(description='Test Install')
    parser.add_argument('--config', help="pretrained model config" , required=True, type=str)
    parser.add_argument('--checkpoints', help="pretrained model checkpoint weights" , required=True, type=str)
    parser.add_argument('--image', help="test image" , required=True, type=str)
    args = parser.parse_args()
    return args

args = parse_args()

config_file = args.config
checkpoint_file = args.checkpoints

register_all_modules()

# switch to cuda if necessary
model = init_detector(config_file, checkpoint_file, device='cpu')

image = mmcv.imread(args.image,channel_order='rgb')
result = inference_detector(model, image)

from mmdet.registry import VISUALIZERS
visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

visualizer.add_datasample('result',image,data_sample=result,draw_gt = None,wait_time=0,)

visualizer.show()