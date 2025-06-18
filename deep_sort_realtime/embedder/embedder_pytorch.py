import os
import logging

import cv2
import numpy as np
import pkg_resources
import torch
from torchvision.transforms import transforms

from deep_sort_realtime.embedder.mobilenetv2_bottle import MobileNetV2_bottle

logger = logging.getLogger(__name__)

MOBILENETV2_BOTTLENECK_WTS = pkg_resources.resource_filename(
    "deep_sort_realtime", "embedder/weights/mobilenetv2_bottleneck_wts.pt"
)

TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS = pkg_resources.resource_filename(
    "deep_sort_realtime", "embedder/weights/osnet_ain_ms_d_c_wtsonly.pth"
)

INPUT_WIDTH = 224


def batch(iterable, bs=1):
    l = len(iterable)
    for ndx in range(0, l, bs):
        yield iterable[ndx : min(ndx + bs, l)]


class MobileNetv2_Embedder(object):
    """
    MobileNetv2_Embedder loads a Mobilenetv2 pretrained on Imagenet1000, with classification layer removed, exposing the bottleneck layer, outputing a feature of size 1280.

    Params
    ------
    - model_wts_path (optional, str) : path to mobilenetv2 model weights, defaults to the model file in ./mobilenetv2
    - half (optional, Bool) : boolean flag to use half precision or not, defaults to True
    - max_batch_size (optional, int) : max batch size for embedder, defaults to 16
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    - gpu (optional, Bool) : boolean flag indicating if gpu is enabled or not
    """

    def __init__(
        self, model_wts_path=None, half=True, max_batch_size=16, bgr=True, gpu=True
    ):
        if model_wts_path is None:
            model_wts_path = MOBILENETV2_BOTTLENECK_WTS
        assert os.path.exists(
            model_wts_path
        ), f"Mobilenetv2 model path {model_wts_path} does not exists!"
        self.model = MobileNetV2_bottle(input_size=INPUT_WIDTH, width_mult=1.0)
        self.model.load_state_dict(torch.load(model_wts_path))

        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            self.model.cuda()  # loads model to gpu
            self.half = half
            if self.half:
                self.model.half()
        else:
            self.half = False

        self.model.eval()  # inference mode, deactivates dropout layers

        self.max_batch_size = max_batch_size
        self.bgr = bgr

        logger.info("MobileNetV2 Embedder for Deep Sort initialised")
        logger.info(f"- gpu enabled: {self.gpu}")
        logger.info(f"- half precision: {self.half}")
        logger.info(f"- max batch size: {self.max_batch_size}")
        logger.info(f"- expects BGR: {self.bgr}")

        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup

    def preprocess(self, np_image):
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        """
        功能​：检查输入图像是否为 BGR 格式（通过 self.bgr 标志控制），如果是，则通过切片 [..., ::-1] 将通道顺序从 BGR 翻转为 RGB；否则，直接使用原图像。
        意义​：OpenCV 读取的图像默认为 BGR，而大多数深度学习模型（如 PyTorch 预训练模型）要求 RGB 格式。此步骤确保颜色空间一致性，避免模型因通道顺序错误而性能下降。
        """
        if self.bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image

        """
        功能​：使用 OpenCV 的 resize 函数将图像缩放至固定尺寸 (INPUT_WIDTH, INPUT_WIDTH)。
        意义​：统一图像尺寸是预处理的关键步骤，能保证所有输入具有相同空间维度，便于批量处理和模型计算。INPUT_WIDTH 是预设常量（如 224），通常选择标准大小（如 224×224）以兼容常见网络架构（如 ResNet）
        """
        input_image = cv2.resize(np_image_rgb, (INPUT_WIDTH, INPUT_WIDTH))
        """
        ​功能​：定义一个转换组合 trans，包含两个步骤：

        transforms.ToTensor()：将 NumPy 数组或 PIL 图像转换为 PyTorch 张量（数据类型从 uint8 转为 float32），并自动将通道维度置于最前（形状从 H × W × C 变为 C × H × W）。
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])：对张量进行归一化，减去均值并除以标准差。

        意义​：Compose 将多个预处理操作串联，提升代码效率。归一化使用 ImageNet 数据集的统计参数，将像素值从原始范围 [0, 255] 映射到近似标准正态分布（约 [-1, 1]），加速模型收敛并提高泛化能力。
        """
        trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        """
        功能​：应用上一步定义的转换组合 trans 到图像上，依次执行 ToTensor() 和 Normalize()。
        ​意义​：此步完成图像到张量的核心转换，并确保数据分布符合模型预期。归一化后，张量值范围约为 [-2.5, 2.5]，避免了梯度爆炸或消失问题
        """
        input_image = trans(input_image)
        """
        功能​：使用 PyTorch 的 view 方法重塑张量维度，从 (3, INPUT_WIDTH, INPUT_WIDTH) 变为 (1, 3, INPUT_WIDTH, INPUT_WIDTH)。
        ​意义​：添加批次维度（大小为 1）是因为 PyTorch 模型要求输入为四维张量 (batch_size, channels, height, width)。即使处理单张图像，也需显式添加批次维度，以满足网络前向传播的输入要求
        """
        input_image = input_image.view(1, 3, INPUT_WIDTH, INPUT_WIDTH)

        return input_image

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 1280)

        """
        all_feats = []
        """
        ​调用 preprocess​：对每张图像执行：

        颜色空间转换（BGR→RGB）
        尺寸统一调整为 (INPUT_WIDTH, INPUT_WIDTH)
        转换为 PyTorch 张量 + ImageNet 标准化
        输出形状：(1, 3, INPUT_WIDTH, INPUT_WIDTH)

        ​结果​：生成预处理后的张量列表 preproc_imgs
        """
        preproc_imgs = [self.preprocess(img) for img in np_images]

        for this_batch in batch(preproc_imgs, bs=self.max_batch_size):
            # 将批次内的多个单图像张量（形状 (1, C, H, W)）沿第 0 维拼接
            this_batch = torch.cat(this_batch, dim=0)
            if self.gpu:
                this_batch = this_batch.cuda()
                if self.half:
                    this_batch = this_batch.half()
            # 将批次数据输入嵌入模型
            output = self.model.forward(this_batch)
            # 特征回收与转换
            all_feats.extend(output.cpu().data.numpy())

        return all_feats


class TorchReID_Embedder(object):
    """
    Embedder that works with torchreid (https://github.com/KaiyangZhou/deep-person-reid). Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

    Params
    ------
    - model_name (optional, str): name of model, see torchreid model zoo. defaults to osnet_ain_x1_0 
    - model_wts_path (optional, str) : path to torchreid model weights, defaults to TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS if model_name=='osnet_ain_x1_0' (default) and else, imagenet pretrained weights of given model 
    - bgr (optional, Bool) : boolean flag indicating if input frames are bgr or not, defaults to True
    - gpu (optional, Bool) : boolean flag indicating if gpu is enabled or not
    - max_batch_size: Does nothing, just for compatibility to other embedder classes
    """

    def __init__(
        self, model_name=None, model_wts_path=None, bgr=True, gpu=True, max_batch_size=None,
    ):
        try: 
            import torchreid 
        except ImportError: 
            raise Exception('ImportError: torchreid is not installed, please install and try again or choose another embedder')
        
        from torchreid.utils import FeatureExtractor
        
        if model_name is None: 
            model_name = 'osnet_ain_x1_0'

        if model_wts_path is None: 
            model_wts_path = ''

        if model_name=='osnet_ain_x1_0' and model_wts_path=='':
            model_wts_path = TORCHREID_OSNET_AIN_X1_0_MS_D_C_WTS

        self.gpu = gpu and torch.cuda.is_available()
        if self.gpu:
            device = 'cuda'
        else:
            device = 'cpu'

        self.model = FeatureExtractor(
            model_name=model_name, 
            model_path=model_wts_path,
            device=device,
        )

        self.bgr = bgr

        logger.info("TorchReID Embedder for Deep Sort initialised")
        logger.info(f"- gpu enabled: {self.gpu}")
        logger.info(f"- expects BGR: {self.bgr}")

        zeros = np.zeros((100, 100, 3), dtype=np.uint8)
        self.predict([zeros])  # warmup

    def preprocess(self, np_image):
        """
        Preprocessing for embedder network: Flips BGR to RGB, resize, convert to torch tensor, normalise with imagenet mean and variance, reshape. Note: input image yet to be loaded to GPU through tensor.cuda()

        Parameters
        ----------
        np_image : ndarray
            (H x W x C)

        Returns
        -------
        Torch Tensor

        """
        if self.bgr:
            np_image_rgb = np_image[..., ::-1]
        else:
            np_image_rgb = np_image
        # torchreid handles the rest of the preprocessing
        return np_image_rgb

    def predict(self, np_images):
        """
        batch inference

        Params
        ------
        np_images : list of ndarray
            list of (H x W x C), bgr or rgb according to self.bgr

        Returns
        ------
        list of features (np.array with dim = 1280)

        """
        preproc_imgs = [self.preprocess(img) for img in np_images]
        output =  self.model(preproc_imgs)
        return output.cpu().data.numpy()
