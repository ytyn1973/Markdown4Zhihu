# yolact算法及其实现

## yolact简介  

  yolact是一种一阶段的实例分割方法，它将分割任务划分为两个并行的子任务，其一是生成一组原型掩码（Prototypes），其二是生成一组预测参数（Mask Coefficients），两者组合处理形成一个mask。这两个子任务前有一些预处理（骨干网，FPN）以获得特征图（Feature Map），生成的结果也要经过非极大抑制（NMS），组合后还要经过裁剪（Crop）和阈值化（Threshold）才能得到最后的结果。这种算法因Prototypes使用原图尺度，不像Rol pooling那样损失精度，且并行计算使得其与一些二阶段（如Mask-R-Cnn）相比在精度损失不大的情况下速度快很多。本文将参考 [论文](https://arxiv.org/pdf/1904.02689)和[代码](https://github.com/dbolya/yolact)，主要依照如下流程图的结构介绍。
![总流程图](https://raw.githubusercontent.com/ytyn1973/Markdown4Zhihu/master/Data/yolact/总流程图.png)

## 算法细节
*注：yolact的官方仓库在实现是往往提供了多种方案，比如骨干网输入支持550,400，300等多种尺寸，backbone有ResNet，DarkNet，VGG多种实现，在卷积上提供可变性卷积等不同的参数类型，对于特征图有一些可选的小网络进行进一步特征提取，权重也有一些可选的项如Coefficient Diversity Loss，Mask IoU Loss。本文对于可选的地方都只介绍其中一种。拓展功能少部分介绍。* 

**1.骨干网提取**

对于不同尺度的图片输入，通过resize和padding将其统一放缩到（550*550）的尺度便于后续处理。

骨干网和核心结构是这样的

```python
x = conv1 → bn1 → relu → maxpool

for layer in self.layers:
    x = layer(x)
    outs.append(x)
```

先过一层普通卷积+BN+Relu+最大池化作为网络头部，循环的每一层包含多个残差块（Bottleneck），四层的输出分别对于C2, C3, C4, C5。这是一个典型的Resnet。假设输入的尺度H×W×3。其尺度和通道数的变化如图所示。
| 名称 | 来自哪  | 通道数 | 尺度 |
|------|--------|--------|------|
| C1   | conv1  | 64     | H/2 × W/2 |
| C2   | layer1 | 256    | H/4 × W/4 |
| C3   | layer2 | 512    | H/8 × W/8 |
| C4   | layer3 | 1024   | H/16 × W/16 |
| C5   | layer4 | 2048   | H/32 × W/32 |

这些输出将作为FPN的输入

**2.FPN**

在骨干网提取特征图时，随着feature map尺度不断变小，语义信息不断提取的同时细节信息也在不断丢失。并且检测小尺度特征的能力也在不断丢失，想要兼顾对大小尺度特征的识别，就需要用综合利用不同尺度的feature map。而骨干网得到的C2-C5虽然是不同尺度，但是语义层级和通道数不统一，难以直接使用，因此通过上、下采样和特征图相加构建P3-P7的特征金字塔。

具体怎么做呢？我们将C5进行1*1卷积统一通道数得到P5，将P5上采样的结果与C4对应相加得到P4，P3也是同理。而P6，P7则是通过P5下采样获得。P3到P7的尺度与大小如表所示:
| 名称 | 通道数 | 尺度 |
|------|--------|------|
| P3   | 256    | H/8 × W/8 |
| P4   | 256    | H/16 × W/16 |
| P5   | 256    | H/32 × W/32 |
| P6   | 256    | H/64 × W/64 |
| P7   | 256    | H/128 × W/128 |

**3.Prototypes**

对于每一张图，都要通过Protonet生成K个共享的mask原型。通常将FPN的到的P3作为Protonet的输入,用中高层语义特征生成Protype masks往往比用原图生成好。其输入如下：

``proto_x: [B, 256, H, W]       #B为批次，H,W为原图的1/8``

在进入Protonet前，还会将一个固定的预先生成的坐标网格作为额外通道加到proto_x中，使得具有平移不变性的CNN对位置更加敏感，便于mask生成。


Protonet是一个简单的三层卷积，得到的结果会经过激活函数做归一处理（yolact此处用的是rule，这意味着prototype都是非负的，这对后面的设计有影响），同时对维度次序进行调整便于后面乘系数。中间的结果还可以保存一份作为 prediction head部分网络的输入以提升效果。注意这里的输出尺寸是小于原图尺寸的，最后要得到原图上的mask还要进行上采样等后处理。

`` proto_out: [B, H, W, K] ``

prototype的数量k并不会很大，一方面是是因为数据集往往有着相似的特征，另一方面是分割的结果还会按 bbox crop修正，也会有补偿。K一般取24。可以把得到prototype视为向量空间一组维度为K的基，基与系数相乘得到向量，对应prototype与另一分支得到的系数进行矩阵乘法得到实例mask。

**4.Prediction Heads**


FPN得到的P3-P7每个层都会进行PredictionModule处理。对于同一张图片的不同层他们会共享卷积。PredictionModule里有三个分支，分别用来预测边框（bbox）类别（conf）和mask系数

由于不同层负责不同尺度的目标，因此每一层的预选框大小也是不同的，yolact中的默认值是[24, 48, 96, 192, 384]，分布比较适中。如果自己训练时数据集上的特征尺度分布比较极端，也可以相应调整以适应。

```python
self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)
self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)
```

他们的输入都是一样的，这些网络的主要作用不是提取特征（他们的输入已经是特征图了）而是把特征图映射成预测结果。

bbox_layer确定每个anchor需要(dx, dy, dw, dh)四个参数，因此输出通道数就是
```
[B, num_priors*4, H, W]
```
conf_layer要预测种类，每个通道对应的就是一种种类的概率，因此输出通道就是
```
[B, num_priors*num_classes, H, W]
```
mask_layer要预测系数，因为prototype是K维的，而mask是系数与prototype的线性组合，因此每个anchor需要K个系数。因此输出通道就是
```
[B, num_priors*K, H, W]
```

因为prototype除了相加还得支持相减，而构造prototype用的是rule，全是正的。所以这里的激活函数用tanh而非rule。

以下为这部分的示意图

![分支](https://raw.githubusercontent.com/ytyn1973/Markdown4Zhihu/master/Data/yolact/分支.png)

在实际实现的时候会把prior也放在输出里，但这并不意味着prior也是网络生成的，而是因为网格预测的是``Δx, Δy, Δw, Δh``，还需要prior才能计算出真正的bbox用来计算loss和一些后处理。注意这三个检测头的结果是一一对应的。对应的一组结果会进行拼接以便后续处理。

**5.后处理**

- 解码bbox：把网络预测的偏离量结合prior得到box的坐标
- 置信度过滤：对每个类别，去掉低分框，只保留score高于阈值的候选框
- 按类别做NMS（box NMS）：对每个类别按score排序，计算IOU，将IOU大于阈值的删掉


经过这几步后得到的是少量优质的final_boxes，final_scores，final_classes，final_mask_coeff.

- 生成实例 mask：根据公式

$$M_i(x, y) = \sum_{k} c_{ik} \cdot P_k(x, y)$$


将基mask和mask 系数线性组合，其中 $cik$ 表示第i个检测框的第k个prototype的权重。$Pk（x，y）$表示第k个prototype在$（x，y）$处的值。
- 按bbox裁剪mask：此时得到的mask是全图范围内的，而实例分割对于某一实例只关注框内。所以需要按照bbox的区域，将区域外的裁剪掉。
- resize：由于prototype是依据P5得到的，其大小显然小于原图，线性组合后的结果也小于原图，需要通过resize使其与原图一致
- 二值化：此时生成的mask实际上代表该点属于实例的概率，因此要有一个阈值进行二值化，这样就得到了最后的mask

**6.Train**

之前介绍的是前向推理过程，对于训练好的模型进行单张图片推理的时候就是这个流程。在训练中这个流程会发生一些变化，比如NMS就不做了，为了得到更多的训练数据。但是总体上每张图还是会有类似的的推理过程。所以训练部分主要介绍loss函数的计算。

损失主要有三部分，分别是边框损失，类别损失和mask损失
$$
L_{\text{total}} = L_{\text{bbox}} + L_{\text{conf}} + L_{\text{mask}}
$$

- 边框损失：Yolact中的边框损失用的是Smooth L1 Loss
$$L_{\text{loc}} = \sum_{i \in \text{正样本}} \sum_{j \in \{x, y, w, h\}} \text{SmoothL1}(\hat{b}_{i,j} - b_{i,j})$$

$\hat{b}_{i,j}$是预测的偏移量，$b_{i,j}$是匹配的ground truth边框，这里只计算正样本的损失。其中Smooth L1
$$\text{SmoothL1}(x) =
\begin{cases} 
0.5 x^2, & |x| < 1 \\
|x| - 0.5, & |x| \ge 1
\end{cases}$$

Smooth L1在偏差小时取平方，损失大时取线性。使得整体损失比较平滑且不会梯度过大。

由于边框损失只针对正样本，以上的公式会有这样的问题：预测大量的框，蒙对了血赚，蒙错了没损失。因此要进行归一化并乘以权重（N为正样本数量）
$$L_{\text{loc}} = \frac{\alpha}{N} \sum_{i \in \text{正样本}} \text{SmoothL1}(\hat{b}_i - b_i)$$

- 分类损失：yolact对于分类损失提供了多种方案，这里以（普通交叉熵 + OHEM）为例。交叉熵的计算方法为
$$L_{\mathrm{CE}} = - \frac{1}{N} \sum_{i=1}^{N} \log \Big( \mathrm{softmax}(\mathbf{c}_i)_{y_i} \Big)$$
但是由于负样本数量远远大于正样本，因此这里要用OHEM（Online Hard Example Mining）。计算负样本的loss score
$$s_i = \log \Big( \sum_{c=0}^{C-1} e^{c_i^c} \Big) - c_i^0$$
取最大的K个保证对抗样本数量相当

- mask损失
公式如下
$$L_{\text{mask}} = \frac{\alpha}{N_{\text{pos}}} \sum_{i=1}^{N_{\text{pos}}} \text{BCE}(M_i, GT_i)$$
其中GT为真实mask，M为组合且裁剪后的预测mask。BCE是对每个像素点计算损失
 $$\text{BCE}(M_i, GT_i) = - \sum_{x,y} \Big[ GT_i(x,y) \log M_i(x,y) + (1 - GT_i(x,y)) \log (1 - M_i(x,y)) \Big]$$

**7.测试**
按照仓库中的教程配置环境和下载权重后可进行测试
```
python eval.py --trained_model=weights/yolact_darknet53_54_800000.pth --score_threshold=0.15 --top_k=15 --image=my_image.png
```
测试了两张图片如下：

![大象测试](https://raw.githubusercontent.com/ytyn1973/Markdown4Zhihu/master/Data/yolact/大象测试.png)
![菠萝测试](https://raw.githubusercontent.com/ytyn1973/Markdown4Zhihu/master/Data/yolact/菠萝测试.png)


可见对于coco数据集中有的类别模型能识别的很好，否则就难以识别，这也体现了yolact作为语义分割依然要预测bbox和类别的原因

接下来试着在自己的数据集上训练，首先要下载数据集，以[TACO](https://github.com/pedropro/TACO)数据集为例，这是一个垃圾分类的数据集。github下载的到的是json文件和图片对于的url。可以通过下载到的脚本下载图片，官方提供的down.py是串行下载的，可以改成并行的加快速度（但是依然很慢）
