from mmengine.visualization import Visualizer
import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import glob
import argparse
from mmengine import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Image Inference')
    parser.add_argument('--config', help="pretrained model config" , required=True, type=str)
    parser.add_argument('--checkpoint', help="pretrained model checkpoint weights" , required=True, type=str)
    parser.add_argument('--image', help="test image" , required=True, type=str)
    args = parser.parse_args()
    return args

args = parse_args()

cfg = Config.fromfile(args.config)

checkpoint_file = glob.glob(args.checkpoint)[0]

img = mmcv.imread(args.image,channel_order='rgb')

model = init_detector(cfg, checkpoint_file, device='cuda:0')

new_result = inference_detector(model, img)

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

visualizer.dataset_meta = model.dataset_meta

visualizer.add_datasample('new_result', img, data_sample=new_result, draw_gt=False, wait_time=0, out_file=None, pred_score_thr=0.5)

visualizer.show()