mIoU达到89.01的超级参数：
lr=0.1, batch-size=8

# TODO:
重新设计click train data获取方式，独立出一个文件供pascal调用<br>
对当前代码进行重构。<br>
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
