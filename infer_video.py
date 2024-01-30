import argparse
import mmcv
import cv2
import time
from mmdet.apis import init_detector, inference_detector
import glob
from mmengine import Config
from mmdet.registry import VISUALIZERS
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Video Inference')
    parser.add_argument('--config', help="pretrained model config" , required=True, type=str)
    parser.add_argument('--checkpoint', help="pretrained model checkpoint weights" , required=True, type=str)
    parser.add_argument('--video', help="test video" , required=True, type=str)
    parser.add_argument('--output', help="output" , required=True, type=str)
    args = parser.parse_args()
    return args

args = parse_args()

cfg = Config.fromfile(args.config)
checkpoint_file = glob.glob(args.checkpoint)[0]

model = init_detector(cfg, checkpoint_file, device='cuda:0') # !should change to cuda! complie torch thru cuda!!!

cap = mmcv.VideoReader(args.video)
save_name = args.output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(
    f"data/{save_name}.mp4", fourcc, cap.fps,
    (cap.width, cap.height)
)
frame_count = 0 
total_fps = 0

visualizer = VISUALIZERS.build(model.cfg.visualizer)
visualizer.dataset_meta = model.dataset_meta

for frame in cap:
    frame_count += 1
    start_time = time.time()# start time.
    result = inference_detector(model, frame)
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    total_fps += fps
    
    visualizer.add_datasample('result',frame, data_sample=result, draw_gt = None,wait_time=0,)
    
    show_result = visualizer.get_image()
    
    cv2.putText(
        show_result, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
        1, (0, 0, 255), 2, cv2.LINE_AA
    )
    out.write(show_result)

out.release()
cv2.destroyAllWindows()
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")