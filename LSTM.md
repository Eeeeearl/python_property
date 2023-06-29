### 代码整理

---

#### 数据准备

##### 预处理

数据归一化：

1、最值归一化

这个适合边界明确的数据，知道最大值最小值的界限。
$$
x_{new} = \frac{x - x_{min}}{x_{max} - x_{min}}
$$
2、均值方差归一化

适用性优于最值归一化，一般是建议优先使用这个归一化手段。将数据归一化为均值为0，方差为1的数据集。
$$
s = \frac{1}{n}\sum^n_{i=1}(x_i - \bar{x}), x_{new} = \frac{x - \bar{x}}{s}
$$


```python
dim = 4  # 单个数据维度
SEQLEN = 10  # 用10个数据去预测  （前10分钟的数据预测下一个数据）
# 定义数据格式
x_train = np.zeros((dataLen-1084-1431-100, SEQLEN, dim))
y_train = np.zeros((dataLen-1084-1431-100, dim))
# 按时间序列创建数据，并输入输出一一对应
for j in range(SEQLEN, total_data.shape[0]):
    y_train[j-SEQLEN] = total_data[j]
    x_train[j-SEQLEN] = total_data[(j-SEQLEN): j]  # 输入数据SEQLEN个一组

```



#### 模型定义

```python
from keras.layers import LSTM, Dense
from keras.models import Sequential

# 线性网络的架构，框架
model = Sequential()
# LSTM参数：50：输出空间的维数；
# 		   input_shape：输入格式；
#		   activation：激活函数；
# 		   recurrent_dropout：动态变化的丢弃参数dropout
#          return_sequences：一般LSTM进行多个连接使用，最后一个设置成False
model.add(LSTM(50, input_shape=(SEQLEN, dim), activation='relu', recurrent_dropout=0.01))
# Dense 线性全连接层；dim：输出维数；activation：激活；
model.add(Dense(dim, activation='linear'))
# keras中定义优化器：adam；损失函数：mae（均方根误差）；来进行反向传播的计算
model.compile(loss='mae', optimizer='adam')
```

> 关于dropout方面：
>
> 1、以下这种dropout，是对LSTM的输出层进行随机丢弃；
>
> ```python
> model.add(LSTM(10))
> model.add(Dropout(0.5))
> ```
>
> 2、而LSTM内部的dropout分为两个参数：dropout 和 recurrent_dropout
>
> ```python
> model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
> ```
>
> dropout：输入x和隐藏层hidden之间的；
>
> recurrent_dropout：hidden与hidden之间的；

> 模型评估标准
>
> （1）准确率(Accuracy)
>
> 本文所研究的基于深度学习技术的疫情预测方法，输入为疫情相关的历史数据，输出为未来疫情趋势的预测值。实验首选的模型评估指标为“感染者数量”的准确度，即在单位时间内或者某一时间点上，模型预测感染新型冠状病毒的患者人数与真实人数的比值。
>
> （2）均方预测误差（MSFE）
>
> 均方预测误差（MSFE）用于测量预测变量对特定值的预测与真实值之间的期望平方距离，是对预测变量质量的度量。其主要表示的意义是预测变量和估计变量之间的差异。
>
> （3）平均绝对百分比误差（MAPE）
>
> 平均绝对百分比误差（MAPE）是对预测系统的准确性的统计度量。它以百分比形式测量此准确性，其可以表示为每个时间段内的平均绝对百分比误差与实际值之差除以实际值，平均绝对百分比误差（MAPE）是用于预测误差的最常用度量，在数据没有极端数据值和没有零值存在的时候，其评估的效果是最好的。
>
> （4）R2分数(r2 score)
>
> R平方（R2）是一个统计指标，表示回归模型中由一个或多个自变量解释的因变量变化的比例。相关性解释了自变量和因变量之间的关系强度，而R平方解释了一个变量的方差在多大程度上可以对第二个变量的方差进行解释。



#### 模型训练

```python
history = model.fit(x_train, y_train, epochs=200, batch_size=72, validation_data=(x_test, y_test), verbose=2, shuffle=False)
```

> - verbose本意为冗长的
>
>   verbose=True/False表示是/否展示详细信息
>
>   model.fit 中的 verbose
>   verbose：日志显示
>   verbose = 0 为不在标准输出流输出日志信息
>   verbose = 1 为输出进度条记录
>   verbose = 2 为每个epoch输出一行记录
>   注意： 默认为 1
>
> - shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。

```python
for step in range(200):
    cost = model.train_on_batch(x_train, y_train)
    # 每50轮，打印一次cost值
    if step % 50 == 0:
        print(f'cost:{cost}')

# 打印权值和偏置值
W, b = model.layers[0].get_weights()
print('W：', W, ' b: ', b)
print(len(model.layers))
```





#### 模型保存与加载

**第一种**

最简单的保存，只保存模型的参数（**需要自己去重新创建模型**）

```python
# 保存模型
model.save_weights('./checkpoint')

# 加载模型
model = create_model()
model.load_weights('./checkpoint')

# 评估模型
loss, acc = model.evaluate(X, Y)
```



**第二种**

保存模型的所有参数（**目前推荐使用这个**）

```python
# 保存模型
model.save('model.h5')

# 加载模型
model = keras.models.load_model('model.h5')

# 评估模型
model.evaluate(X, Y)

# 模型预测
model.predict(X)
```

> 这边加载保存的内容有：
>
> - 模型的结构，允许重新创建模型
> - 模型的权重
> - 训练配置项（loss，optimizer）
> - 优化器状态，允许准确地从你上次结束的地方继续训练



**第三种**

得到的模型可以直接部署，不需要代码给用户（工业环境部署，暂未验证）

```python
#保存模型
tf.saved_model.save(m,'tmp/saved_model/')
 
#导出模型
imported = tf.saved_model.load(path)
f = imported.signatures['serving_default']
```





