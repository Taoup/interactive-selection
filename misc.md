



训练记录：<br>
click-net 没有头尾shortcut连接效果更好：0.8141 VS 0.8254<br>
PFM方式的网络，在训练800epoch后，mIoU达到0.8213, 70左右epoch时大概为0.75样子<br>
在已经训练的分割网络上进行sbox的训练，第一个epoch的mIoU就可以达到0.70样子。<br>
sbox_on_deeplab:训练deeplab的所有参数VS只训练decoder:***对所有参数进行训练，训练5个epoch后，val
集mIOU达到0.8454。结论：训练所有参数。*** 
sbox_on_deeplab后续训练出现train loss持续上升的状态，可能原因：数值稳定性问题，某个
batch的gradient炸了。可以说试试gradient norm clipping.
![](abnormal_training.PNG)



# TODO:
~~resume 训练后，learning rate 比结束训练前升高很多，导致resume后很长一段时间都没有提升。~~<br>
------------使用训练参数--ft即可<br> 
~~sbox使用已训练的分割网络进行训练，使用较好的学习率调整策略. be patient.~~<br>
~~更精细的learning rate schedual~~<br>
~~加上sbox的jitter~~<br>
~~在预处理中去掉太小的目标物体~~<br>

### Segmentation中处理255
Loss函数中有个ignore_index


### 将图像转换为5通道的关键代码
```python
In [65]: (2,) + img.shape[1:]
Out[65]: (2, 270, 360)

In [66]: zz = np.zeros((2,)+img.shape[1:])

In [67]: zz.shape
Out[67]: (2, 270, 360)

In [68]: tt = np.append(img, zz, axis=0)

In [69]: tt.shape
Out[69]: (5, 270, 360)
```

### 模拟用户输入
    max filter and min filter to abtain potitive candidate and
    negative candidate
    

### Euclidian distance map
```python
xs = np.array(range(img.shape[1]))
ys = np.array(range(img.shape[2]))

xss = (xs-seed[1]) **2
yss = (ys-seed(0])) ** 2

euclid = np.array([np.sqrt(x+y) for x in xss for y in yss]).reshape(seed).astype(np.int32)
euclid[euclid > 255] = 255
stack = np.stack([euclid, euclid2], axis = 0)                                                                                                                  

In [40]: fused_map = np.min(stack, axis=0)            
```
