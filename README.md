# Fine-tuning RTMDet for Instance Segmentation

Welcome to Makeability Lab's repository about fine-tuning RTMDet models! Fine-tuning is a pivotal process in deep learning where a pre-trained model, already trained on a large dataset, is further trained or "fine-tuned" on a smaller, specific dataset. This approach leverages the learned features and patterns from the initial training, making it highly efficient for tasks like image classification, object detection, and more in CV.

In this repo we mainly applies the approach of feature extraction, in which we freeze the base layers of the model, leveraging their learned features, and only train the final layers specific to our tasks.

Kindly note that this repo is valid as of Jan 29th 2024, in any future circumstances where OpenMMLab, owner of mmdetection and rtmdet, changes their implementation, please refer to their official [github](https://github.com/open-mmlab/mmdetection).

## Setup

Firstly you will have to setup conda environment, mmdetection toolbox and pytorch. If your GPU supports CUDA, please also install it.

1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html).
1. Install [Pytorch](https://pytorch.org/get-started/locally/). Choose CUDA version when you have CUDA device.
1. Install [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html).

It is recommended that you first install Pytorch and then mmdetection otherwise your Pytorch might not be correctly complied with CUDA.

## Fine-Tuning

1. Once you installed everything, firstly make three folders inside the mmdetection directory namely `./data`, `./checkpoints` and `./work_dir` either manually or using `mkdir` in conda.

1. The next step is to download pre-trained config and weights files from mmdetection. For example, `mim download mmdet --config rtmdet-ins_l_8xb32-300e_coco --dest ./checkpoints`. This means that this is a pre-trained RTMDet instance segmentation model that has been trained on a COCO dataset with 8 GPUs, a batch size of 32 and 300 epochs. You should name your weights file in the same way and you can find all config files for all available models [here](https://github.com/open-mmlab/mmdetection/tree/main/configs).

1. After downloading the pre-trained model that you would like to work with, run `test_install.py` to see if it is working correctly. If you can see an image with segmentation masks pops out, then you have installed everthing correctly. Otherwise check the error messages and google.

1. Move your COCO_MMdetection dataset to `./data` and run `coco_classcheck.py` to check the classes contained in your data.

1. To fine-tune a pre-trained model, you will have to setup a customized config file. Check and run `config_setup.py`.

1. Now, run `tools/train.py PATH/TO/CONFIG` and let the training process start. If in any circumstances the training is interrupted but the last checkpoint is successfully saved into './work_dir', you can resume the process from the where it stopped by running `tools/train.py PATH/TO/CONFIG --resume auto`. Remember to toggle the resume option in your config file to the `True`.

1. When training is done, run `infer_img.py` or `infer_video.py` to test the fine-tuned model on either a single image or a video.