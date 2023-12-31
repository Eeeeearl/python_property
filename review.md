#### 传统目标检测~深度目标检测

**思路**：

通常利用滑动窗口进行选取目标候选框，然后再利用一些算法进行特征提取，最后输入分类器进行检测分类。

通常效率较低，准确度只能说还可以。

---

#### RCNN

Region-base Convolutional Neural Network 或者 Region with CNN

**关键点**：

 1）成功将深度模型应用在了目标检测上，属于开创性模型；

 2）引入“Region”概念（区域）；

**步骤**：

 1）对图像进行提取若干兴趣区域；

 2）CNN 抽取特征；

 3）对每个区域进行类别和边界框的预测回归。

> 选择性搜索 -> 图片 -> 卷积神经网络 -> 类别预测&边界框预测

**缺点**：

 1）计算量过大，需要对图片进行划分N多个框去预测，增加网络的前向计算。

#### Fast R-CNN

较快的 R-CNN

**关键点**：

 1）对图片直接进行卷积操作；

 2）引入了“兴趣区域池化层”（Region of interest pooling layer，ROI池化层）

**解决R-CNN 瓶颈**————对每个Region进行单独的特征提取（独立的特征提取会导致大量的重复计算）；

方法：对整个图片先进行卷积，再进行兴趣区域池化。

**步骤**：

 1）对图片整体做卷积操作；（输出：$1×c×h_1×w_1$）

 2）将1）的输出和选择性搜索生成的n个Region信息（形状相同的特征，$h_2,w_2$），作为ROI池化层的输入，输出连接后的各个Region的兴趣区域所抽取的特征；（输出：$n×c×h_2×w_2$）

 3）再经过全连接层；（输出：$n×d$）

 4）预测类别，形状变为$n×q$并使用Softmax回归（q为类别数）；预测边界框，形状变为$n×4$（4记录边界框信息）；

> 图片 -> 卷积神经网络 -> ROI池化 -> 全连接层 -> 类别预测&边界框预测
>
> ↓------> 选择性搜索 ------↑

> ROI池化层可以理解为类似于Max Pooling层

#### Faster R-CNN

更快的R-CNN

Faster R-CNN 在 Fast R-CNN 的基础上，对选择性搜索进行了改进，Fast R-CNN需要在选择性搜索中生成较多的Region，目标检测结果才会较为精确；而Faster R-CNN将其替换为**区域提议网络（RPN，Region Proposal Network）**，减少了Region数量，并保证了精度。

**关键点**：

 1）区域提议网络（RPN，Region Proposal Network）；主要工作：让网络可以同时学习到如何生成高质量的Region（提议区域），这是减少Region数量也保证精度的关键所在。

> -> 卷积神经网络 -> **卷积层 -> 锚框 -> 二元类别预测&边界框预测 -> 非极大值抑制NMS** -> ROI池化 -> 

#### Mask R-CNN

它则是考虑了每个目标在图像上的像素级位置，通过更精确的标注，以得到更高的目标检测精度。

**思路**：

 1）将ROI池化层替换成**兴趣区域对其层**；（通过双线性插值来保留特征图上的空间信息，以适合像素级预测）

> -> 兴趣区域对其层 -> 全连接层 -> 类别预测&边界框预测
>
> ​         ↓-------> 全卷积网络 -> 掩码预测（像素级）

---

#### CNN和FCN

**CNN**：

image **--conv->** feature map **--pool->** feature map **--FC Flatten->** vector **--FC->** vector **--FC->** result

**FCN**:

image **--conv->** feature map **--pool->** feature map **--conv->** feature map **--conv->** feature map **--conv->** result

不过FCN包含了三个**特点**：

1）全卷积网络；（fully convolutional）将原本 CNN 的最后三层全连接层变成卷积层，可以适应任意尺寸的输入，也大幅减少计算量，降低模型复杂度。 

2）卷积层上采样；（upsample）采用反卷积网络，使用双线性插值方法完成上采样过程。

3）**跳层结构**；（skip layer）直接利用上采样对最后一层特征图进行上采样的话，会损失很多细节，边缘模糊，使用跳跃结构，将最后一层的预测（有丰富的全局信息）和更浅层（有更多的局部细节）的预测结合起来（sum 方式），可以恢复细节信息。 

> 跳层结构，在此认为应该是语义分割网络的核心，yolo后续系列中，也都有他的身影。
>
> 结合Resnet网络的结构特点，感觉<u>深层浅层的特征相融合</u>对功能实现还是具备良好效果。

**FCN对CNN的改进**

FCN对图像进行像素级的分类，从而解决了语义级别的图像分割（semantic segmentation）问题。FCN把cnn全连接层换成了卷积层，可以接受任意尺寸的输入图像，采用反卷积对卷积层的特征图进行采样，使它恢复到输入图像相同的尺寸，从而可以对每个像素都产生了一个预测，同时保留了原始输入图像的空间信息，然后在上采样上的特征图进行逐像素分类，最后逐个像素计算softmax损失，相当于每个像素对应一个训练样本。 

> 14年到17年的语义分割模型：
>
> - 2014年 FCN 模型，主要贡献为在语义分割问题中推广使用端对端卷积神经网络，使用反卷积进行上采样
> - 2015年 U-net 模型，构建了一套完整 的编码解码器
> - 2015年 SegNet 模型，将最大池化转换为解码器来提高分辨率
> - 2015年 Dilated Convolutions（空洞卷积），更广范围内提高了内容的聚合并不降低分辨率
> - 2016年 DeepLab v1&v2 （该系列将==空洞卷积==能力发挥出来）
> - 2016年 RefineNet 使用残差连接，降低了内存使用量，提高了模块间的特征融合
> - 2016年 PSPNet 模型
> - 2017年 Large Kernel Matters
> - 2017年 DeepLab V3

---

#### 语义分割

看两个内容：空洞卷积、DeepLabv3（v1-v3的优化路径）

**空洞卷积**

Dilated Convolution

定义：在卷积核元素之间添加空格（零）来扩大卷积核的过程叫做空洞卷积。

也叫作扩张卷积或者膨胀卷积。

意义：在卷积扩张率为1时，其实就是常规的卷积操作，但修改扩张率后，可以在不增加参数量的前提下增加视野范围。因此，扩张卷积可用于廉价地增大输出单元的感受野，而不会增大其核大小，这在多个扩张卷积彼此堆叠时尤其有效。<u>但也是需要进行系统性地聚合信息，以免丢失分辨率或者其他内容</u>。

> U-net因为网络结构中存在大量的池化层来进行下采样，大量使用池化层的结构就是会丢失掉一些信息，因此在重构分辨率的时候会有些吃力，而且在多目标、小目标的语义分割上，以U-Net为代表的分割模型存在精度瓶颈。
>
> 在此背景下，以空洞卷积为重大创新的DeepLab系列网络出现。

> 感受野概念重提：
>
> 所谓感受野，是指输出特征图上某个像素对应到输入空间中的区域范围。所以感受野可以理解为特征图像素到输入区域的映射。相对而言，越处于感受野中间的像素越重要。

总结：其三个主要作用：

1. 扩大感受野；
2. 获取多尺度上下文信息；（当多个带有不同dilation rate的空洞卷积核叠加时，不同的感受野会带来多尺度信息，这对于分割任务是非常重要的。） 
3. 参数量没有额外增加；



**DeepLab**

由Google团队提出，2014年开始陆续提出v1、v2、v3、v3+等版本。

==DeepLabv1==

**创新点**：

- 空洞卷积；Dilated Convolution
- 全连接条件随机场 DenseCRF（Conditional Random Field）
- 多尺度预测（这应该不算做创新点，反正就是用到了）

**算法流程**：

input -> Deep Convolution Neural Network -> Coarse Score map（粗糙的得分map） -> Bi-linear Interpolation（双线性插值） -> Fully Connected CRF -> Final Output

> CRF 简单来说就是在决定每一个位置的像素值时，会考虑周围的像素值。（==属于后处理部分，不参与训练的==）
>
> 在传统图像处理上是做一个平滑的处理操作。
>
> 那在语义分割上面就不友好，语义分割希望的是恩怨分明，有锐化信息，这样才能有精确的图像轮廓。

1. 全连接条件随机场 DenseCRF：是认为图像中每个像素点都与其他所有像素点相关，为每个像素都建立了pairwise potential（成对的可能性）。（个人理解：==大概就是当前像素点和其他像素点进行的一个语义分析==）
2. 空洞卷积Dilated Convolution：略
3. 多尺度预测，也就是在整个卷积过程中，在多个Pooling层进行预测，不同的pooling层则有不同尺度的预测效果。

在这里多尺度预测没有DenseCRF那么惊艳，但也有一定的效果，所以模型最终结合了DenseCRF和Multi-scale prediction。

![DeepLabv1网络结构](./img/deeplabv1.jpg)

作者在vgg16的基础上进行改进，为了加载预先在ImageNet训练好的VGG-16模型，并保证图片仅缩放了8倍做了如下修改：

- 把全连接层（fc6、fc7、fc8）改成卷积层（做分割嘛）
- 把最后两个池化层（pool4、pool5）的步长2改成1（保证feature的分辨率）
- 把最后三个卷积层（conv5_1、conv5_2、conv5_3）的dilate rate设置为2，且第一个全连接层的dilate rate设置为4（保持感受野）
- 把最后一个全连接层fc8的通道数从1000改为21（分类数为21）

---

==DeepLabv2==

与v1对比，v2将==原本的VGG网络更改成了Resnet==，并==增加了ASPP结构==，用于解决不同检测目标大小差异的问题：<u>通过在给定的特征层上使用不同扩张率的空洞卷积，ASPP可以有效地进行重采样</u>。

1. ASPP，Atrous Spatial Pyramid Pooling，空洞空间金字塔池化。

空间金字塔池化的概念最早是在目标检测的经典算法SPP-Net（Yolo里面也用到了）中提出。

**核心思想**是聚集不同尺寸的感受野，是用来解决不同分割目标不同尺寸的问题。

![Atrous Spatial Pyramid Pooling示意图](./img/deeplabv2-ASPP.jpg)

根据扩张率的设置不同，提出了两种不同尺度的ASPP：ASPP-S和ASPP-L，对应的扩张率：{2,4,8,12}{6，12,18,24}，在DeepLabv2中，Pool5之后的空洞卷积替换成了ASPP，就是设置四个对应的扩张率，进行操作，然后相加，得到最终结果。

---

==DeepLabv3==

1. 将DeepLabv1&v2的全连接CRF移除；
2. 引入Multi-Grid策略，即多次使用空洞卷积而不是只使用一次；
3. 优化ASPP结构，例如加入BN层；

**Multi-Grid策略**

思想：在一个block中连续使用多个不同扩张率的空洞卷积。

> 其是为了解决空洞卷积可能产生的gridding问题（网格问题）。
>
> 因为在扩张率变大后，对采集的信息会变得稀疏，造成局部信息的丢失，导致长距离上一些语义不相关的信息；也就是网格化后，信息方面可能不连贯了。

> 图像网格化的原因：
>
> 1. 连续使用了扩张率相同的空洞卷积

![普通空洞卷积vsHDC](./img/空洞卷积使用.jpg)

**优化ASPP**

作者发现，随着空洞卷积的扩张率增大，卷积核中的有效权重越来越少（那确实嘛，空洞本身就减少了权重参数的使用，自身携带的优点）；实验表明，当扩张率达到一定程度，就退化成了1x1卷积。

那为了解决这个问题，DeepLabv3参考了ParseNet的思想，==增加了一个由来提升图像的全局视野的分支==。

具体的说，它先使用GAP将Feature Map的分辨率压缩至 1×1 ，再使用 1×1 卷积将通道数调整为 256 ，最后再经过**BN以及双线性插值上采样**将图像的分辨率调整到目标分辨率。因为插值之前的尺寸是 1×1 ，所以这里的双线性插值也就是简单的像素复制。

---

==DeepLabv3+==

DeepLabv3中没有考虑到浅层的特征，因此借鉴了FPN等网络的encoder-decoder架构，实现了feature map跨block的融合，提出了DeepLabv3+版本。

**创新点**：

1. 采用了encoder-decoder架构；
2. 使用了分组卷积来加速；

**encoder-decoder架构**

原v3版本是将ASPP模块得到的feature map结果进行1x1的分类层后直接双线性插值到原始图片大小，这个过于暴力。

> 双线性插值——bilinear interpolation
>
> > 线性插值：大概可以这么理解，就是给了你两个点的坐标，便可以确定一条直线，那么给你一个x坐标，就可以得到它的类似y坐标，这边是线性插值。
> >
> > 最近邻法(Nearest Interpolation)：
> >
> > ![](./img/最近邻插值.jpg)
> >
> > 原图像直接对应位置进行复制过来，其他的值通过最接近的数值进行填充。
> >
> > 特点：不用计算——速度快；但破坏了新图像中像素的渐变关系。
>
> 双线性插值：大概就是在两个方向上进行线性插值，首先会先确定四个点位置，然后上下各两个点先进行x方向确定两个坐标值，再利用这两个值再进行y方向的线性插值，得到最终结果。
>
> 四个点哪来？
>
> 一般来说resize后的值来自于resize前原图像该点（浮点）四周的整点的值，因此一般x2 =x1+1,x1 = floor(x),y2=y1+1,y1 =floor(y) 
>
> 应用：比如图像的**resize**操作。

DeepLab v3+使用DeepLab v3作为Encoder，我们重点关注它的解码器模块。它分成7步： 

1. 首先我们先通过编码器将输入图像的尺寸减小16倍；
2. 使用 1×1 卷积将通道数减小为 256 ，后再接一个BN，ReLU激活函数和Dropout；
3. 使用双线性插值对对齐进行上采样 4 倍；
4. 将缩放$4$倍处的浅层的特征依次经过 1×1 卷积将通道数减小为 48 ，BN，ReLU；
5. 拼接3和4的Feature Map；
6. 经过两组 3×3 卷积，BN，ReLU，Dropout；
7. 上采样4倍得到最终的结果。

![DeepLab v3+的Encoder-Decoder网络结构](./img/DeepLabv3+.jpg)

---

当然还有很多其他的优化处理，这里直接对DeepLab系列进行个简单了解，就先整理这么多。

---



#### 目标跟踪检测

这块内容主要面向视频检测，只有视频流这样的数据才有必要进行目标跟踪。

大部分可以用来对车辆、行人等，场景大概就是在公路上、商场里或者其他一些特定场合。

目标跟踪，顾名思义就是实现对某个单一目标实现持续定位，进而知道他的一个运动轨迹。

**IOU跟踪**

就是利用IOU框来进行目标定位，大概就是通过目标检测得到目标框，视频流在不同时间点得到不同目标框，当前时间点，则利用上一时间点的结果进行IOU对比，选择出IOU最大的，就认为它们是同一目标。

很显然，会造成不良问题——目标丢失：比如两个人穿插经过、经过障碍物等，就很容易造成目标混乱，跟踪失败。当然不能上来就解决很难的问题，得感受下算法发展的历程，感受算法的魅力。

**卡尔曼滤波**

<u>卡尔曼滤波 Kalam Filter 适合用于线性的状态处理</u>。

> 差不多就是他可以根据当前的状态去预测下一次的状态。

用在目标跟踪上的话，就可以对上面的IOU跟踪进行优化 --> <u>利用卡尔曼滤波去预测下目标的下一个位置，然后再结合当前检测结果来进行IOU计算，从而实现跟踪</u>。

![](./img/卡尔曼公式图示.png)

上图就是卡尔曼的状态方程的图解。
$$
状态方程 x_k = Ax_{k-1} + Bu_k +w_k
$$

$$
观测方程 y_k = Cx_{k} + v_k
$$

这里的$w_k,v_k$属于噪声，$u_k$表示状态的一个变化量。

> 暂时看到的内容没有把我解释通，反正大概意思就是使用上一次的结果，来推测当前得值，然后用观测值去修正，得到最后结果；这也就是训练ABC等参数的过程。



**sort算法——实现多目标跟踪**

卡尔曼滤波算是优化了最初的IOU方案，但这只是单一目标，若存在多目标去跟踪，显然乏力。

sort算法进行进一步优化。

为顺利目标跟踪，sort算法包含两部分：

- 匈牙利算法——为了确定谁是谁；
- 卡尔曼滤波——为了预测物体状态以便进行跟踪；

sort算法作者的出发点仅仅是**检测框的位置和大小**进行**目标的运动估计和数据关联**，并没有去用到目标的外观特征，也没有进行其他重识别的算法，都是**关注于帧与帧之间的匹配**。

<u>但是SORT算法只适用于遮挡情况少的、运动比较稳定的对象</u>。 

*匈牙利算法*

匈牙利算法（Hungarian algorithm）：解决二分图最大匹配问题的算法。

> 二分图：就是被划分为两个部分的特殊的图；
>
> 例这边划分两个集合A、B，同一集合中的任意两点不会存在连线，只有和另一个集合中的点才能相连。

![](./img/匈牙利算法-图例.png)

实现结果：在现有基础上，实现最优匹配结果。

上图结果：(Box1， Target2),(Box2， Target3),(Box3， Target1) 



**deepsort**

sort算法在遇到有障碍物的时候，还是会容易丢失跟踪目标。

deepsort在sort基础上增加了**级联匹配**（Matching Cascade）和**新轨迹的确认**（confirmed）；使用了CNN网络提取特征，增加了对丢失和遮挡的鲁棒性，高效适用于在线场景。

轨道Tracks 分为两种状态：确认态（confirmed）、不确认态（unconfirmed）。

新产生的Tracks是不确定态的；只有它和Detection 连续匹配一定次数（默认为3）才会转为确定态。而确定态的Tracks 必须和Detection 连续失配一定次数（默认为30），才会被删除（移除确定态）。

![](./img/deepsort-流程图.png)

整个算法的工作流程：

1. 第一帧检测到的结果创建其对应的Tracks（unconfirmed状态）。将卡尔曼滤波的运动变量初始化，通过卡尔曼滤波预测其对应的box。

2. 将该帧目标检测box 和上一帧通过Tracks预测的box 一一进行IOU匹配，再通过IOU 匹配的结果计算其代价矩阵（cost matrix，其计算方式为 1-IOU）。

3. 将2. 中得到的所有的代价矩阵作为匈牙利算法的输入，得到线性的匹配的结果，这时候我们得到有三种结果：

   1. **Unmatched Tracks**（这里需要判断一下，他是不是确定态的，若是，则记录连续失配次数，满足要求直接删除；若不是，则直接进行删除。）
   2. **Unmatched Detections**（Detections 失配，会将这样的Detections 初始化成一个新的Tracks。）
   3. **Matched Tracks**（代表检测框和预测框匹配成功 => 说明前一帧与后一帧追踪成功，将其对应的Detections 通过卡尔曼滤波更新其对应的Tracks 变量。）

4. 反复循环2. -3. 两个步骤，直到出现确认态confirmed 的Tracks 或者视频帧结束。

5. 通过卡尔曼滤波预测其确认态的Tracks 和不确认态的Tracks 对应的box，将确认态的Tracks 的box 和Detections进行级联匹配（Matching Cascade）。

   > 之前每次只要Tracks 匹配上都会保存Detections 的 **外观特征** 和 **运动信息**，默认保存前100帧，利用 **外观特征和运动信息** 和 **Detections** 进行 **级联匹配**，这么做是因为确认态的Tracks和Detections匹配的可能性更大。

6. 进行级联匹配后同样会有三种结果：

   > 1和2 代表的是Tracks 和Detections 不匹配，这时要将之前的不确认态的Tracks 和失配的 Tracks 一起和 Unmatched Detections 一一进行IOU匹配，再通过IOU 匹配的结果计算其代价矩阵（cost matrix，其计算方式为 1-IOU）

   1. **Unmatched Tracks**
   2. **Unmatched Detections**
   3. **Matched Tracks**（Tracks 匹配上了，这样的Tracks 要通过卡尔曼滤波更新其对应的Tracks 变量。）

7. 将6. 得到的所有的代价矩阵作为匈牙利算法的输入，得到线性的匹配的结果，这时候我们得到有三种结果：

   1. **Unmatched Tracks**（这里需要判断一下，他是不是确定态的，若是，则记录连续失配次数，满足要求直接删除；若不是，则直接进行删除。）
   2. **Unmatched Detections**（Detections 失配，会将这样的Detections 初始化成一个新的Tracks。）
   3. **Matched Tracks**（代表检测框和预测框匹配成功 => 说明前一帧与后一帧追踪成功，将其对应的Detections 通过卡尔曼滤波更新其对应的Tracks 变量。）

   <u>（7. 同上面第3. 内容一致）</u>

8. 反复循环5. -7. 步骤，直到视频帧结束。



**deepsort + yolov5**

实现大致流程如下：

![](./img/yolov5+deepsort-图示.png)

我们是需要自行训练yolo 和deepsort 权重的。

> yolo 跟踪检测系列，有不同版本。
>
> https://github.com/mikel-brostrom/yolov8_tracking/tree/v8.0

