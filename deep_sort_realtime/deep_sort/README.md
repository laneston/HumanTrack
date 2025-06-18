





# 余弦距离计算公式变量详解

## 公式定义:


$$cosinedistance=1-\cos{θ}=1-\frac{a⋅b}{∥a∥∥b∥}$$

其中：

- $​​a$ 和 $b​$​：待比较的两个向量，可以是文本特征向量、图像像素向量等。
- $​​a⋅b​$​：向量点积（内积），计算方式为各维度分量乘积之和，即


$$a⋅b=\sum_{i=1}^{n}{{{a}_{i⋅}{b}_{i}}}$$


$∥a∥$ $∥b∥$ ：向量的模长（L2范数），计算公式为：

$$∥a∥=\sqrt{\sum_{i=1}^{n}{{{a}_{i}^{2}}}}$$


​​cos(θ)​​：向量夹角的余弦值，反映方向一致性，值域为 [−1,1]

## 关键点解析

​向量标准化​：若向量未归一化，模长会影响结果。归一化后公式简化为：
$cosinedistance=1−{a}_{norm}⋅{b}_{norm}$
其中${a}_{norm}=\frac{a}{∥a∥}$


## 关键代码解析


```
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
```

`np.asarray(a)`将输入数据 a 转换为NumPy数组（ndarray）。若 a 本身是NumPy数组且满足数据类型和存储顺序要求，则直接返回原数组（无复制开销）。


`np.linalg.norm(a, axis=1, keepdims=True)`计算数组的L2范数（默认）。此处指定 axis=1，表示对每行计算范数；keepdims=True 保持结果维度与输入一致，便于广播操作。


```
1.0 - np.dot(a, b.T)
```

功能​：计算矩阵乘法或向量内积，其中 b.T 表示对矩阵 b 进行转置。a 和 b 必须是 numpy 数组。a 的列数需等于 b.T 的行数（即 b 的列数）


# 欧氏距离

## ​定义与公式​

欧氏距离（Euclidean Distance）是欧几里得空间中两点之间的直线距离，其数学原理基于勾股定理的推广。对于n维空间中的两个点 $P({p}_{1},{p}_{2},...,{p}_{n})$ 和 $Q({q}_{1},{q}_{2},...,{q}_{n})$
，其欧氏距离公式为：

$$d(P,Q)=\sqrt{{\sum_{i=1}^{n}{{({{p}_{i}}-{{q}_{i}})}^{2}}}}$$

## 代码解析

```
    a, b = np.asarray(a), np.asarray(b)
```

- ​作用​：将输入对象 a 和 b 强制转换为 NumPy 数组。
- ​函数解析​：`np.asarray` 可将列表、元组等数组类对象转换为 `ndarray`。若输入已为NumPy数组且满足 `dtype` 和 `order` 要求，​不复制数据，直接返回原数组引用，否则创建新数组。
- ​意义​：统一数据结构，便于后续矩阵运算。


```
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
```

- ​作用​：检测输入数组是否为空，若为空则返回全零矩阵。
- ​解析​：
  - `len(a)` 获取数组行数（若为二维数组）。
  - 避免后续计算因空输入导致异常，确保代码鲁棒性。




```        
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
```

- 函数解析​：
  - np.square(x)：对数组元素逐项平方（数学意义：消除负值影响）。
  - .sum(axis=1)：沿行方向求和，得到每个样本的特征平方和​（如样本向量模长平方）。
- ​输出​：
  - a2：形状 `(M,)`，表示 a 的每行（样本）的平方和。
  - b2：形状 `(N,)`，表示 b 的每行（样本）的平方和。



```
    r2 = -2.0 * np.dot(a, b.T) + a2[:, None] + b2[None, :]
```


$${{∥{{a}_{i}}-{{b}_{j}}∥}^{2}}=\sum_{k=1}^{d}{{{({{a}_{ik}}-{{b}_{jk}})}^{2}}}=-2⋅{{a}_{i}}⋅{{b}_{j}^{T}}+{∥{{a}_{i}}∥}^{2}+{∥{{b}_{i}}∥}^{2}$$

- ​函数解析​：
  - `np.dot(a, b.T)`：计算 a 和 b 行向量的点积矩阵，形状 (M, N)。
  - `a2[:, None]` 和 `b2[None, :]`：通过广播机制将一维数组扩展为 `(M, 1)` 和 `(1, N)`，使相加时自动对齐为 `(M, N)` 矩阵。
- ​意义​：通过矩阵运算避免显式循环，显著提升计算效率（时间复杂度从 O(MN) 优化至矩阵操作）。



```
    r2 = np.clip(r2, 0.0, float(np.inf))
```


- 作用​：将矩阵中的负值截断为0，确保欧氏距离平方非负。
- ​函数解析​： `np.clip` 将输入数组元素限制在 `[0.0, +∞)` 范围内，避免因浮点误差导致的微小负值影响后续操作（如开平方）。
- ​意义​：增强数值稳定性，符合欧氏距离的物理意义。


# 计算单个目标框与多个候选框的交并比（IoU）

```
def iou(bbox, candidates):
    """Computer intersection over union."""
```

- bbox：目标框，格式为(左上角x, 左上角y, 宽度, 高度)
- candidates：候选框矩阵，每行格式与bbox相同

## ​步骤1：坐标转换​


```
bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
candidates_tl = candidates[:, :2]
candidates_br = candidates[:, :2] + candidates[:, 2:]
```

​作用​：将宽高格式转换为绝对坐标形式（左上角+右下角）
`bbox_tl`：目标框左上角坐标 (x1, y1)
`bbox_br`：目标框右下角坐标 (x1 + w, y1 + h)
`candidates_tl/candidates_br`：候选框的左上/右下坐标矩阵

## 步骤2：计算交集区域

```
tl = np.c_[
    np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
    np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis],
]
br = np.c_[
    np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
    np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis],
]
wh = np.maximum(0.0, br - tl)
```


1. ​交集左上角​：取两个框左上坐标的较大值（`np.maximum`）
2. ​交集右下角​：取两个框右下坐标的较小值（`np.minimum`）
3. ​维度处理​：通过`np.c_`和`np.newaxis`对齐矩阵维度，支持批量计算
4. ​有效性检查​：若`br - tl`为负数（无交集），则置为0（`np.maximum(0.0, ...)`）


## 步骤3：面积计算

```
area_intersection = wh.prod(axis=1)
area_bbox = bbox[2:].prod()
area_candidates = candidates[:, 2:].prod(axis=1)
```

- ​交集面积​：交集区域的宽高乘积（`prod(axis=1)`按行求积）
- ​目标框面积​：宽度 × 高度
- ​候选框面积​：每个候选框的宽度 × 高度

## 步骤4：IoU计算

```
return area_intersection / (area_bbox + area_candidates - area_intersection)
```

- ​公式​：IoU = 交集面积 / (目标框面积 + 候选框面积 - 交集面积)
- ​特性​：
  - 结果范围在[0, 1]之间，值越大表示重叠度越高
  - 支持批量处理（一次计算多个候选框的IoU）



# 计算跟踪目标（tracks）与检测框（detections）之间的IoU代价矩阵


## 定义与计算原理

### 基本概念

IoU（交并比）衡量两个边界框的重叠程度，通俗来说，并交比指的是两个矩形框相交的面积，比上两个矩形框相并的面积，IoU值越大说明定位越准确，最大值为1。计算公式为：IoU=交集面积/并集面积

​IoU代价矩阵则通过将IoU转化为匹配成本，具体形式为：

$$Cost(i,j)=1-IoU({D}_{i},{T}_{j})$$

其中，${D}_{i}$ 为第i个检测框，${T}_{j}$ 为第j个跟踪轨迹的预测框。

### 构建规则

矩阵的行对应当前帧的检测框，列对应上一帧的跟踪框（或预测框）。

每个元素${C}_{kj}$​表示检测框 $k$ 与跟踪框 $j$ 的IoU代价







```
def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """An intersection over union distance metric."""
```

- 功能​：计算跟踪目标（tracks）与检测框（detections）之间的IoU代价矩阵，用于目标跟踪中的关联匹配（如DeepSORT算法）。
-  参数​：
   -  tracks：跟踪器维护的轨迹列表，每个轨迹包含状态（如位置、速度）和更新信息。
   -  detections：当前帧的检测框列表，每个检测框包含坐标（如ltwh格式）和置信度。
   -  track_indices：需匹配的轨迹索引（默认全部）。
   -  detection_indices：需匹配的检测框索引（默认全部）。

## 步骤1：索引初始化

```
if track_indices is None:
    track_indices = np.arange(len(tracks))
if detection_indices is None:
    detection_indices = np.arange(len(detections))
```

- ​作用​：若未指定索引，默认选择所有轨迹和检测框参与计算。
- ​原理​：生成从0到N-1的连续索引，覆盖所有可能匹配项。


## 步骤2：代价矩阵初始化

```
cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
candidates = np.asarray([detections[i].ltwh for i in detection_indices])
```

- 关键变量​：
  - cost_matrix：初始化为全零矩阵，形状为 (轨迹数, 检测数)。
  - candidates：提取检测框的坐标（ltwh格式，即 [左上x, 左上y, 宽, 高]）。
- ​技巧​：直接提取检测框坐标，避免后续循环中重复访问对象属性。


## 步骤3：遍历轨迹计算代价

```
for row, track_idx in enumerate(track_indices):
    if tracks[track_idx].time_since_update > 1:
        cost_matrix[row, :] = linear_assignment.INFTY_COST
        continue
    bbox = tracks[track_idx].to_ltwh()
    cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
```

- ​逻辑分解​：
  - ​失效轨迹处理​：若轨迹超过1帧未更新（time_since_update > 1），将其代价设为极大值（INFTY_COST），表示不参与匹配。
  - ​坐标转换​：将轨迹的状态（如Kalman滤波结果）转换为ltwh格式，与检测框格式一致。
  - ​IoU计算​：调用iou()函数（需预先定义），计算当前轨迹与所有检测框的IoU值，再通过 1 - IoU 转换为代价（IoU越大，代价越小）。
- ​关键函数​：
  - iou(bbox, candidates)：计算单个轨迹框与多个检测框的IoU，返回一维数组。

## ​步骤4：返回代价矩阵​

```
return cost_matrix
```

​输出​：形状为 (M, N) 的矩阵，用于匈牙利算法等匹配方法，寻找最优轨迹-检测关联。


# NearestNeighborDistanceMetric

这个 `NearestNeighborDistanceMetric` 类用于多目标跟踪中的特征距离计算，维护目标特征库并生成匹配成本矩阵。管理每个目标（跟踪对象）的特征样本，计算新特征与目标之间的最小距离，用于数据关联（如目标匹配）。

## 初始化方法 __init__

```
def __init__(self, metric, matching_threshold, budget=None):
    # 选择距离度量函数
    if metric == "euclidean":
        self._metric = _nn_euclidean_distance  # 欧氏距离（最近邻）
    elif metric == "cosine":
        self._metric = _nn_cosine_distance     # 余弦距离（最近邻）
    else:
        raise ValueError("Invalid metric")
    # 设置匹配阈值和预算
    self.matching_threshold = matching_threshold
    self.budget = budget
    self.samples = {}  # 存储目标特征：{target_id: [feature1, feature2, ...]}
```

- 参数​：
  - metric: 距离度量方式（"euclidean"或"cosine"）。
  - matching_threshold: 有效匹配的最大允许距离，超过此值的匹配被拒绝。
  - budget: 每个目标保留的最大特征样本数（控制内存和计算量）。
- ​作用​：初始化距离函数、阈值和特征库。

## 更新特征库方法 partial_fit

```
def partial_fit(self, features, targets, active_targets):
    for feature, target in zip(features, targets):
        # 将特征添加到对应目标的样本列表
        self.samples.setdefault(target, []).append(feature)
        # 应用预算限制，保留最近budget个样本
        if self.budget is not None:
            self.samples[target] = self.samples[target][-self.budget :]
    # 仅保留活跃目标，删除非活跃目标的特征
    self.samples = {k: self.samples[k] for k in active_targets}
```

- ​参数​：
  - features: 当前帧的检测特征矩阵（形状N×M，N为检测数，M为特征维度）。
  - targets: 每个检测对应的目标ID（与features一一对应）。
  - active_targets: 当前活跃的目标ID列表（未出现在此列表中的目标将被删除）。
- ​作用​：更新目标特征库，保留最新样本并过滤非活跃目标。

## 距离计算方法 distance

```
def distance(self, features, targets):
    cost_matrix = np.zeros((len(targets), len(features)))
    for i, target in enumerate(targets):
        # 计算目标特征与当前检测特征的最小距离
        cost_matrix[i, :] = self._metric(self.samples[target], features)
    return cost_matrix
```

- ​参数​：
  - features: 待匹配的检测特征矩阵（形状K×M，K为检测数）。
  - targets: 需要计算距离的目标ID列表。
- ​返回​：成本矩阵（形状len(targets)×len(features)），元素(i,j)表示目标targets[i]与检测features[j]的最小距离。
- ​作用​：生成关联成本矩阵，用于匈牙利算法等匹配方法。



# Track 类

## TrackState 状态

```
    Tentative = 1  # 初始暂定状态，需积累足够匹配次数
    Confirmed = 2  # 确认状态，持续跟踪中
    Deleted = 3  # 删除状态，需从跟踪列表移除
```

## 初始化

```
        # 状态变量
        self.mean = mean  # 卡尔曼滤波的均值向量（中心坐标x,y，宽高比a，高度h）
        self.covariance = covariance  # 卡尔曼滤波的协方差矩阵
        self.track_id = track_id  # 唯一ID
        self.hits = 1  # 成功匹配的次数
        self.age = 1  # 轨迹存在的总帧数
        self.time_since_update = 0  # 距上次更新的帧数
        # 状态管理，可能内存增长，需限制长度或使用滑动窗口
        self.state = TrackState.Tentative  # 初始为暂定状态
        self.features = []  # 特征缓存（用于外观模型匹配）
        self.latest_feature = None  # 最新特征
        if feature is not None:
            self.features.append(feature)
            self.latest_feature = feature
        # 参数配置，需根据场景调整，如高帧率场景可增大max_age
        self._n_init = n_init  # 确认所需连续匹配次数
        self._max_age = max_age  # 最大允许失配帧数
        # 关联的检测信息
        self.original_ltwh = original_ltwh  # 原始检测框坐标（左上宽高）
        self.det_class = det_class  # 检测类别
        self.det_conf = det_conf  # 检测置信度
        self.instance_mask = instance_mask  # 实例分割掩码
        self.others = others  # 其他自定义数据
```


## 坐标转换方法

```
to_ltwh(orig=False, orig_strict=False)
```

- ​作用​：将状态均值或原始检测框转为(x,y,w,h)格式。
- ​参数​：
  - orig=True：优先返回原始检测框（若无且orig_strict=True则返回None）。
  - orig_strict=True：严格模式，仅返回原始框。
- ​注意​：原始检测框可能在预测后被清空，需在update后立即使用。

```
to_ltrb(orig=False)
```

- ​作用​：将边界框转为(x_min,y_min,x_max,y_max)。
- ​依赖​：先调用to_ltwh获取基础坐标。

## 检测信息获取方法

```
def get_det_conf(self):    # 当前关联检测的置信度（可能为None）
def get_det_class(self):   # 检测类别（持久化，即使无当前检测）
def get_instance_mask(self):  # 实例掩码（可能为None）
```

## 预测与更新

```
predict(kf)
```

- 作用​：卡尔曼滤波预测状态，并更新存活时间。
- ​注意​：需传入与卡尔曼滤波器兼容的kf对象。

```
update(kf, detection)
```

- ​作用​：用新检测更新状态，并记录特征。
- 注意​：detection需包含feature、ltwh等属性。


## 状态管理

```
mark_missed()
```

逻辑​：未匹配时标记，Tentative轨迹直接删除，Confirmed轨迹超时删除。

状态检查方法

作用​：判断当前状态，用于外部管理跟踪列表。

## 使用注意事项

1. ​坐标转换​：
   - 确保to_ltwh和to_ltrb的输入输出与检测器/可视化工具兼容。
   - 使用orig=True时需确认当前帧是否有关联检测（original_ltwh非空）。
2. ​特征管理​：
   - 大量轨迹时features可能占用内存，需定期清理或限制长度。
3. ​卡尔曼滤波器兼容性​：
   - predict和update需与具体卡尔曼实现接口一致。
4. 状态生命周期​：
   - n_init和max_age需根据场景调整（如摄像头帧率、目标运动速度）。
5. 多线程/进程安全​：
   - 若在多线程环境中修改轨迹状态，需加锁保护。
6. ​检测信息时效性​：
   - det_conf、instance_mask等仅在update后有效，predict后会被重置。
7. ​自定义数据​：
   - others字段可存储任意数据，但需注意序列化和访问安全。


# Tracker类

```
class Tracker:
    """多目标跟踪器"""
    def __init__(self, metric, max_iou_distance=0.7, max_age=30, n_init=3, ...):
        # 初始化参数
        self.today = today  # 跟踪ID日期标识
        self.metric = metric  # 外观特征距离度量
        self.max_iou_distance = 0.7  # IOU匹配阈值，高于此值的IOU认为不匹配
        self.max_age = 30  # 最大丢失帧数，根据场景调整，长则轨迹保留更久但可能有噪声
        self.n_init = 3  # 确认轨迹所需连续检测次数，新轨迹需稳定出现3次才确认，防止误检
        self.kf = kalman_filter.KalmanFilter()  # Kalman滤波器
        self.tracks = []  # 活跃轨迹列表
```

- ​参数说明​：
  - metric：用于计算检测与轨迹外观相似度的距离度量（如余弦距离）。
  - max_age：轨迹被删除前的最大丢失帧数。
  - n_init：新轨迹需连续匹配N次检测才会被确认。

 ## predict()

- ​作用​：对每个轨迹进行Kalman滤波预测，预测目标位置和速度，状态向量​：通常包含 [x, y, 宽高比, 高度, vx, vy, vh]。
- ​注意​：需在每个时间步的update()前调用。

## update()

- ​流程​：
  - 日期变更时重置ID。
  - 通过_match()进行检测-轨迹关联。
  - 更新匹配轨迹状态，标记丢失轨迹，初始化新轨迹。
  - 清理无效轨迹，更新外观特征模型。
- ​注意​：detections需包含有效的特征向量。

## _match()

- ​策略​：
  - 优先用级联匹配处理已确认轨迹。
  - 剩余轨迹和检测用IOU匹配。
- ​优势​：兼顾外观特征和运动模型。

- 级联匹配​：优先匹配丢失时间短的轨迹，降低ID切换。
- ​IOU匹配​：作为后备方案，处理外观特征失效的情况。

## _initiate_track()

- 作用​：将未匹配的检测初始化为新轨迹。
- ​ID生成​：包含日期和自增ID，避免重复。


# 卡尔曼滤波器

```
chi2inv95 = {1: 3.8415, 2: 5.9915, ..., 9: 16.919}
```

- ​作用​：存储卡方分布在 0.95 置信水平下的临界值，用于计算马氏距离的门控阈值。
- ​用途​：判断测量值是否属于合理范围（例如，chi2inv95[4] 对应 4 自由度的阈值）。

## 初始化 __init__

```
def __init__(self):
    ndim, dt = 4, 1.0
    self._motion_mat = np.eye(2 * ndim, 2 * ndim)
    for i in range(ndim):
        self._motion_mat[i, ndim + i] = dt
    self._update_mat = np.eye(ndim, 2 * ndim)
    self._std_weight_position = 1.0 / 20
    self._std_weight_velocity = 1.0 / 160
```

- ​作用​：
  - _motion_mat：状态转移矩阵（8x8），实现恒定速度模型（位置 += 速度 × 时间间隔 dt）。
  - _update_mat：观测矩阵（4x8），将 8 维状态映射到 4 维观测空间。
  - _std_weight_*：控制过程噪声和观测噪声的权重。
- ​注意事项​：dt 固定为 1.0，若实际帧率变化需调整。

## 初始化跟踪 initiate

```
def initiate(self, measurement):
    mean_pos = measurement
    mean_vel = np.zeros_like(mean_pos)
    mean = np.r_[mean_pos, mean_vel]
    std = [2 * self._std_weight_position * h, ...]
    covariance = np.diag(np.square(std))
    return mean, covariance
```
- ​作用​：根据测量值初始化跟踪器的均值和协方差。
- ​输入​：measurement 为 (x, y, a, h)。
- ​输出​：8 维均值向量和 8x8 协方差矩阵。
- ​注意事项​：速度初始化为 0，协方差高度依赖 h。

## 预测 predict

```
def predict(self, mean, covariance):
    std_pos = [...]
    std_vel = [...]
    motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
    mean = np.dot(self._motion_mat, mean)
    covariance = FPF^T + motion_cov
    return mean, covariance
```

- ​作用​：预测下一时刻的状态均值和协方差。
- ​输入​：上一时刻的 mean 和 covariance。
- ​输出​：预测后的状态分布。
- ​注意事项​：过程噪声协方差动态依赖于当前状态的 h。

## 投影到观测空间 project

```
def project(self, mean, covariance):
    std = [...]
    innovation_cov = np.diag(np.square(std))
    mean = Hx
    covariance = HPH^T + innovation_cov
    return mean, covariance
```

- ​作用​：将状态分布投影到 4 维观测空间，并添加观测噪声。
- ​输出​：预测的观测均值和协方差。
- ​用途​：用于计算门控距离或更新步骤。

## 更新 update

```
def update(self, mean, covariance, measurement):
    projected_mean, projected_cov = self.project(...)
    kalman_gain = ...  # 使用 Cholesky 分解求解
    innovation = measurement - projected_mean
    new_mean = mean + K * innovation
    new_covariance = covariance - KSK^T
    return new_mean, new_covariance
```

- ​作用​：用测量值修正预测后的状态。
- ​输入​：预测的 mean、covariance 和实际测量值 measurement。
- ​输出​：更新后的状态分布。
- ​注意事项​：依赖数值稳定的 Cholesky 分解。

## 门控距离 gating_distance

```
def gating_distance(self, ...):
    projected_mean, projected_cov = self.project(...)
    cholesky_factor = np.linalg.cholesky(projected_cov)
    z = solve(cholesky_factor, d.T)
    squared_maha = sum(z * z)
    return squared_maha
```

- 作用​：计算测量值与状态分布的马氏距离平方。
- ​输入​：状态分布和多个测量值。
- ​输出​：各测量值的马氏距离平方（需与 chi2inv95 比较）。
- ​注意事项​：若 only_position=True，仅计算中心位置的距离（自由度 2）。






