

### 参数配置和初始化逻辑

```
def __init__(self, max_iou_distance=0.7, max_age=30, ..., polygon=False, today=None):
```

- ​参数说明:
  - ​跟踪参数:
    - max_iou_distance: IoU关联阈值，超过则忽略匹配。
    - max_age: 目标丢失的最大帧数，超时则删除轨迹。
    - n_init: 轨迹初始化所需连续检测帧数。
    - nms_max_overlap: NMS重叠阈值（1.0表示禁用NMS）。
    - max_cosine_distance: 外观特征余弦距离阈值。
    - nn_budget: 外观特征缓存的最大数量。
  - ​模型参数:
    - embedder: 选择特征提取模型（如 mobilenet, clip_ViT-B/16）。
    - half: 是否使用半精度（仅限MobileNet）。
    - bgr: 输入图像是否为BGR格式。
    - embedder_gpu: 是否使用GPU加速特征提取。
  - ​其他:
    - polygon: 是否处理多边形检测（如旋转框）。
    - today: 用于轨迹命名的日期。
- ​核心逻辑:
  - 初始化跟踪器 Tracker 和度量模块 NearestNeighborDistanceMetric。
  - 根据 embedder 参数加载对应模型（MobileNet/TorchReID/CLIP）。
  - 记录初始化配置日志。
- ​注意事项:
  - 使用CLIP模型需提供权重路径 embedder_wts。
  - 多边形检测需设置 polygon=True。

默认初始化`NearestNeighborDistanceMetric`模块为 `cosine` 模式；




### 处理检测结果，更新跟踪器状态

```
def update_tracks(raw_detections, embeds=None, frame=None, ...):
```

- ​输入参数:
  - raw_detections: 原始检测结果，格式取决于 polygon 参数。
  - embeds: 可选预计算的特征向量。
  - frame: 未提供 embeds 时必须提供图像帧。
  - instance_masks: 实例分割掩码（用于优化特征提取）。
- ​处理流程:
  - ​输入校验:
    - 检查 embeds 和 frame 的互斥性。
    - 验证检测框有效性（宽高需>0）。
  - ​特征提取:
    - 若未提供 embeds，调用 generate_embeds 从图像中提取。
    - 多边形检测使用 generate_embeds_poly。
  - ​创建检测对象:
    - 将检测框、置信度、类别和特征封装为 Detection 对象。
  - ​非极大抑制 (NMS)​:
    - 根据 nms_max_overlap 过滤重叠检测。
  - ​跟踪器更新:
    - 调用 tracker.predict() 和 tracker.update() 更新轨迹。
- ​返回:
  - 更新后的轨迹列表 self.tracker.tracks。​
- 注意事项:
  - 输入检测框需为 [left, top, width, height] 或多边形坐标。
  - 实例掩码需与检测框一一对应。

### 辅助方法

- **generate_embeds**: 使用 crop_bb 裁剪检测区域，应用实例掩码（如有），调用嵌入模型提取特征。
- ​**create_detections**: 将原始检测数据封装为 Detection 对象。
- ​**process_polygons**: 将多边形转换为边界矩形（用于特征提取）。
- ​**crop_bb / crop_poly_pad_black**: 裁剪图像区域，支持矩形和多边形。
- ​**refresh_track_ids**: 重置跟踪ID计数器（用于多视频片段处理）。
- ​**delete_all_tracks**: 清空所有跟踪轨迹。


