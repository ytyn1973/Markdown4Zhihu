# yolact算法及其实现

## yolact简介  

  yolact是一种一阶段的实例分割方法，它将分割任务划分为两个并行的子任务，其一是生成一组原型掩码（Prototypes），其二是生成一组预测参数（Mask Coefficients），两者组合处理形成一个mask。这两个子任务前有一些预处理（骨干网，FPN）以获得特征图，生成的结果也要经过非极大抑制（NMS），组合后还要经过裁剪（Crop）和阈值化（THreshold）才能得到最后的结果。这种算法因Prototypes使用原图尺度，不像Rol pooling那样损失精度，且并行计算使得与一些二阶段（如Mask-R-Cnn）相比在精度损失不大的情况下速度快很多。本文将参考论文(https://arxiv.org/pdf/1904.02689)和代码(https://github.com/dbolya/yolact)，主要依照如下流程图的结构介绍。
![总流程图](总流程图.png)

## 算法细节
*注：yolact的官方仓库在实现是往往提供了多种方案，比如骨干网输入支持550,400，300等多种尺寸，backbone有ResNet，DarkNet，VGG多种实现，在卷积上提供可变性卷积等不同的参数类型，对于特征图也有一些可选的小网络进行特征提取。本文对于可选的地方都只介绍其中一种。拓展功少部分介绍。* 

**1.骨干网提取**

对于不同尺度的图片输入，通过resize和padding将其统一放缩到（550*550）的尺度便于后续处理。

骨干网和核心结构是这样的

```python
x = conv1 → bn1 → relu → maxpool

for layer in self.layers:
    x = layer(x)
    outs.append(x)
```

先过一层普通卷积+BN+Relu+最大池化作为网络头部，循环的每一层包含多个残差块（Bottleneck），四层的输出分别对于C2, C3, C4, C5。这是一个典型的Resnet。假设输入的尺度H*W*3。其尺度和通道数的变化如图所示。
| 名称 | 来自哪  | 通道数 | 尺度 |
|------|--------|--------|------|
| C1   | conv1  | 64     | H/2 × W/2 |
| C2   | layer1 | 256    | H/4 × W/4 |
| C3   | layer2 | 512    | H/8 × W/8 |
| C4   | layer3 | 1024   | H/16 × W/16 |
| C5   | layer4 | 2048   | H/32 × W/32 |

这些输出将作为FPN的输入

**2.FPN**

在骨干网提取特征图时，随着feature map尺度不断变小，语义信息不断提取的同时细节信息也在不断丢失。并且检测小尺度特征的能力也在不断丢失，想要兼顾对大小尺度特征的识别，就需要用综合利用不同尺度的feature map。而骨干网得到的C2-C5虽然是不同尺度，但是语义层级，通道数不统一，难以直接使用，因此通过上、下采样和特征图相加构建P3-P7的特征金字塔。

具体怎么做呢？我们将C5进行1*1卷积统一通道数得到P5，将P5上采样的结果与C4对应相加得到P4，P3也是同理。而P6，P6这是通过P5下采样获得。P3到P7的尺度与大小如表所示:
| 名称 | 通道数 | 尺度 |
|------|--------|------|
| P3   | 256    | H/8 × W/8 |
| P4   | 256    | H/16 × W/16 |
| P5   | 256    | H/32 × W/32 |
| P6   | 256    | H/64 × W/64 |
| P7   | 256    | H/128 × W/128 |

**3.Prototypes**

对于每一张图，都要通过Pronet生成K个共享的mask原型。通常将FPN的到的P3作为Pronet的输入,用中高层语义特征开生成Protype masks往往比用原图生成好，所以其尺寸就是P的尺寸。

``proto_x: [B, 256, H, W]       #B为批次，H,W为原图的1/8``

在进入Resnet前，还会将一个固定的预先生成的坐标网格作为额外通道加到proto_x中，使得具有平移不变性的CNN对位置更加敏感便于mask生成。


Pronet是一个简单的三层卷积，得到的结果会经过激活函数做归一处理（yolact此处用的是rule，这意味着prototype都是非负的，这对后面的设计有影响），同时对维度进行调整便于后面乘系数。中间的结果还可以保存一份给 prediction head使用提升那部分网络的效果。注意这里的输出尺寸是小于原图尺寸的，最后要得到原图上的mask还要进行上采样等后处理。

`` proto_out: [B, H, W, K] ``

prototype的数量k并不会很大，一方面是是因为数据集往往有着相似的特征，另一方面是只追求一个近似的表示且且分割的结果还会有按 bbox crop修正，也会有补偿，一般取24。可以把得到prototype视为空间一组维度为K的基，基与系数相乘得到向量。对应prototype与另一分支得到的系数进行矩阵乘法得到实例mask。

**4.Prediction Heads**


FPN得到的P3-P7每个层都会进行PredictionModule。对于同一张图片的不同层他们会贡献卷积的系数。PredictionModule里有三个分支，分别用来预测边框（bbox）类别（conf）和mask系数

由于不同层负责不同尺度的目标，因此每一层的预选框大小也是不同的，yolact中的默认值是[24, 48, 96, 192, 384]，分布比较适中。如何自己训练时数据集上的特征尺度比较极端，也可以相应调整以适应。

```python
self.bbox_layer = nn.Conv2d(out_channels, self.num_priors * 4, **cfg.head_layer_params)
self.conf_layer = nn.Conv2d(out_channels, self.num_priors * self.num_classes, **cfg.head_layer_params)
self.mask_layer = nn.Conv2d(out_channels, self.num_priors * self.mask_dim, **cfg.head_layer_params)
```

他们的输入都是一样的，这些网络的作用不是特征，他们的属于已经是特征图了。而是把特征图的每个位置变成预测结果。

bbox_layer每个anchor需要(dx, dy, dw, dh)四个参数，因此输出通道数就是
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

![分支](分支.png)

在实际实现的时候会把prior也放在输出里，但这并不意味着prior也是网络生成的，而是因为网格预测的是``Δx, Δy, Δw, Δh``，还需要prior才能计算出真正的bbox用来计算loss和一些后处理。注意这三个检测头的结果是一一对应的。

**5.后处理**

- 解码bbox：把网络预测的偏离量结合prior得到box的坐标
- 置信度过滤：对每个类别，去掉低分框，只保留score高于阈值的候选框
- 按类别做NMS（box NMS）：对每个类别按score排序，计算IOU，将IOU大于阈值的删掉


经过这几步后得到的是少量优质的final_boxes，final_scores，final_classes，final_mask_coeff.

- 生成实例 mask：根据公式
$$M_i(x, y) = \sum_{k} c_{ik} \cdot P_k(x, y)$$
将基mask和mask 系数线性组合，其中$cik$表示第i个检测框的第k个prototype的权重。$Pk（x，y）$表示第k个prototype在$（x，y）$处的值。
- 按bbox裁剪mask：此时得到的mask是全图范围内的，而实例分割对于某一实例只关注框内。所以需要按照bbox的区域，将区域外的裁剪掉。
- resize：由于prototype是依据P5得到的，其大小显然小于原图，线性组合后的结果也小于原图，需要通过resize使其与原图一致
- 二值化：此时生成的mask实际上代表该点属于实例的概率，因此要有一个阈值进行二值化，这样就得到了最后的mask