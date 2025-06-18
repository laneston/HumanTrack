## Reference project

- [A really more real-time adaptation of deep sort](https://github.com/levan92/deep_sort_realtime)
- [Introducing Ultralytics YOLO11, the latest version of the acclaimed real-time object detection and image segmentation model. ](https://github.com/ultralytics/ultralytics)




## Dependencies


the torch [download link is here](https://download.pytorch.org/whl/torch/)
the torchvision [download link is here](https://download.pytorch.org/whl/torchvision/)

```
torch==torch-2.3.1 cu118-cp39-cp39-win_amd64.whl
torchvision==torchvision-0.18.1+cu118-cp39-cp39-win_amd64.whl
```

use the following cmd to install torch.

The installation instructions are as follows.

- `pip install '.\torch-2.3.1+cu118-cp39-cp39-win_amd64.whl'`
- `pip install '.\torchvision-0.18.1+cu118-cp39-cp39-win_amd64.whl'`



```
opencv-contrib-python==4.11.0.86
psutil==7.0.0
torchview==0.2.6
```

The installation instructions are as follows.

- numpy, `pip install "numpy<2"`
- opencv, `pip install opencv-contrib-python`
- psutil, `pip install psutil`
- yaml, `pip install PyYAML`
- tqdm, `pip install tqdm`
- requests, `pip install requests`
- pandas,`pip install pandas`
- huggingface_hub, `pip install huggingface_hub`
- scipy, `pip install scipy`
- matplotlib, `pip install matplotlib`

## Tree

```
.
├── LICENSE
├── README.md
├── app
│   └── videocapture.py
├── convert_mpiitoyolo.py
├── deep_sort_realtime
├── models
│   ├── download_yolov10_wts.sh
│   └── yolov11n.pt
├── mpii*
│   ├── images*
│   ├── mpii_human_pose_v1_u12_1.mat*
│   ├── train
│   └── val
└── ultralytics
```

带`*`符号的目录是必备的路径，其余的则由`convert_mpiitoyolo.py`脚本生成

 ## Usage

Enter the app directory by `cd app` and execute instructions `python videocapture.py`.

## 数据集


[You can download the images and annotations from the MPII Human Pose benchmark here:](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download)
