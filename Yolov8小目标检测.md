Yolov8小目标检测，优化踩坑实录

在模型训练过程中，发现两个问题：

1. 在yolov8模型的验证环节，采用自带验证代码计算得到的AP值会与通过COCO数据集官方接口pycocotools计算得到的AP**存在几个百分点的差距**。
2. 在yolov8模型比nanodet模型相比召回率偏低，尤其对于小目标（320\*192输入尺寸10\*10左右）

接下来针对以上两个问题，进行实验分析。

1、通过对比yolov8和pycocotools关于AP值计算的方式进行对比，发现两者的差异在于获取PR曲线采样点时是否采用线性插值的方式。

yolov8采用线性插值：**(utils / metircs.py)**

```python
# Integrate area under curve
method = 'interp'  # methods: 'continuous', 'interp'
if method == 'interp':
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
else:  # 'continuous'
    i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
```

pycocotools采用临近真值：

```python
np.searchsorted(rc, recThrs, side='left')
```

两者为什么会有如此差异？AP的计算需要采样点绘制出PR曲线，共需采样101点。因为PR的原始真值点并不是按照R均匀分布，数量也不固定，所以需要在原始真值点的基础上采样出101个点用于AP计算。

在R依次从0均匀采样到1的过程中，遇到没有对应P真值的情况时，

- yolov8采用的线性插值方式，会寻找左右临近两个真值点通过线性插值计算得到采样点的P值，
- 而pycocotools采用的临近真值方式则是寻找最接近采样点的真值点的P值作为采样点P值。

最后对101个采样得到的P值取均值得到AP。 

如此，AP的差异就可以被发现了，尤其在PR曲线波动较大的类别，两者的结果差距更大。

---

个人觉得，pycocotools采用的临近真值方式会更加合理，因为参与计算的P值都是真值，而yolov8采用的线性插值方式存在估值误差。并且pycocotools作为COCO官方接口，使用更广泛、更具权威性。

针对yolov8中的代码，通过以下修改，可达到和pycocotools相同的结果：

```python
# Integrate area under curve
method = 'interp'  # methods: 'continuous', 'interp'
if method == 'interp':
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
# ADD CONTENT
elif method == 'searchsorted':
    q = np.zeros((102, ))
    x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
    inds = np.searchsorted(recall, x, side='left')
    try:
        for ri, pi in enumerate(inds):
            q[ri] = precision[pi]
    except:
        pass
    q_array = np.array(q)
    ap = np.mean(q_array[q_array > -1])
else:  # 'continuous'
    i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve
```



2、针对小目标召回率低的问题，首先采用和现有优化版nanodet相同策略——增大输出特征图尺寸，使用yolov8n-p2训练。

下图可看出，增大输出特征图尺寸后，yolov8准确率依然占优，整体召回率有明显提升，只有香烟类别召回率依然较nanodet低。

![](./img/yolov8&nanodet准确率.png)

![](./img/yolov8&nanodet召回率.png)

**将yolov8中标签分配方法TAL更换为ATSS后**，香烟召回率和手机AP均有明显提升。虽然相同阈值下，除香烟外其他三类准确率略有下降，但是实际效果影响不大。 



如果大家需要自己绘制上面这些P/R图，我已经集成到pycocotools官方工具中，开源地址：https://github.com/CPFelix/pycocotools。 

---

总结

在对比不同模型的指标时，最好使用同一统计工具。关于AP的计算，还需强调一点的是在横向对比其他模型时，**一定要保证模型输出的置信度阈值和NMS阈值保持一致**。

关于yolov8对于小目标的优化，目前行之有效的策略有两种：**增大输出特征图** 和 **标签分配采用ATSS**。

增大输出特征图很好理解，更大的特征图带来更加精细的底层特征，也意味着更多的候选正样本（落入GT框内），召回率自然会拉升。

而标签分配也是影响最终正样本的重要因素，为什么yolov8使用的TAL会损害小目标的召回呢？

个人看法，欢迎指正：TAL使用模型预测输出的分数和IOU作为度量分数区分正负样本，会存在两个问题：1、冷启动；2、小目标IOU波动大。冷启动的问题在基于模型预测输出区分正负样本的标签分配策略都存在，就是在模型刚开始训练时模型输出是混乱的，正负样本的分配效果也会很差，进而可能导致收敛效果不好。相比中大目标，小目标对于IOU会更加敏感，两个框很小的偏移就会带来IOU很大的差值。两方面因素叠加就会导致最终分配到小目标的正样本偏少，召回率偏低。

另外，在使用ATSS时还可以将锚点框的默认尺寸由5改为3，这样最终生成的默认锚点框会更小，也更适配小目标，会产生更多正样本。

参考文献

[1].https://github.com/ultralytics/ultralytics

[2].https://github.com/RangiLyu/nanodet

[3].https://github.com/CPFelix/pycocotools









































