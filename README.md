# Pytorch-YOLOv4-For Pedestrain Detection

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcuc2hpZWxkcy5pby9zdGF0aWMvdjE?x-oss-process=image/format,png)
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcuc2hpZWxkcy5pby9zdGF0aWMvdjE?x-oss-process=image/format,png)
[![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcuc2hpZWxkcy5pby9zdGF0aWMvdjE?x-oss-process=image/format,png)](./License.txt)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934# Pytorch-YOLOv4-For Pedestrain Detection

![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcuc2hpZWxkcy5pby9zdGF0aWMvdjE?x-oss-process=image/format,png)
![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcuc2hpZWxkcy5pby9zdGF0aWMvdjE?x-oss-process=image/format,png)
[![](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9pbWcuc2hpZWxkcy5pby9zdGF0aWMvdjE?x-oss-process=image/format,png)](./License.txt)

A minimal PyTorch implementation of YOLOv4.
- Paper Yolo v4: https://arxiv.org/abs/2004.10934
- Source code:https://github.com/AlexeyAB/darknet
- More details: http://pjreddie.com/darknet/yolo/


- [x] Inference
- [x] Train
    - [x] Mocaic

```
├── README.md
├── dataset.py            dataset
├── demo.py               demo to run pytorch --> tool/darknet2pytorch
├── demo_darknet2onnx.py  tool to convert into onnx --> tool/darknet2pytorch
├── demo_pytorch2onnx.py  tool to convert into onnx
├── models.py             model for pytorch
├── train.py              train models.py
├── cfg.py                cfg.py for train
├── cfg                   cfg --> darknet2pytorch
├── data            
├── weight                --> darknet2pytorch
├── tool
│   ├── camera.py           a demo camera
│   ├── coco_annotatin.py       coco dataset generator
│   ├── config.py
│   ├── darknet2pytorch.py
│   ├── region_loss.py
│   ├── utils.py
│   └── yolo_layer.py
```
# 模型效果图示    

![seq-1](https://img-blog.csdnimg.cn/20200721204531703.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgyMjYwMA==,size_16,color_FFFFFF,t_70)
![seq-2](https://img-blog.csdnimg.cn/2020072120462061.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgyMjYwMA==,size_16,color_FFFFFF,t_70)
![seq-3](https://img-blog.csdnimg.cn/20200721204618968.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgyMjYwMA==,size_16,color_FFFFFF,t_70)
![seq-4](https://img-blog.csdnimg.cn/20200721204618946.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgyMjYwMA==,size_16,color_FFFFFF,t_70)
![seq-5](https://img-blog.csdnimg.cn/20200721204847443.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MzgyMjYwMA==,size_16,color_FFFFFF,t_70)

# 模型处理视频效果
- google-drive：https://drive.google.com/file/d/1Gn6EnkDNwXMa9lWwPr9kLirEfitNfstB/view?usp=sharing
# 0. Weights Download

## 0.1 darkent
- baidu(https://pan.baidu.com/s/1dAGEW8cm-dqK14TbhhVetA     Extraction code:dm5b)
- google(https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT)

## 0.2 pytorch
you can use darknet2pytorch to convert it yourself, or download my converted model.

- baidu
    - yolov4.pth(https://pan.baidu.com/s/1ZroDvoGScDgtE1ja_QqJVw Extraction code:xrq9) 
    - yolov4.conv.137.pth(https://pan.baidu.com/s/1ovBie4YyVQQoUrC3AY0joA Extraction code:kcel)
- google
    - yolov4.pth(https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
    - yolov4.conv.137.pth(https://drive.google.com/open?id=1fcbR0bWzYfIEdLJPzOsn4R5mlvR6IQyA)
    
## 0.3 Pedestrain-pytorch

-google
    -[YOLOv4-For-Pedestrain.pth](https://drive.google.com/file/d/1-7-vnqQ9EymjTQDdcrLXs9SLdAk0tBjv/view?usp=sharing)



# 1. 代码准备

github 克隆代码
```
git clone https://github.com/BigSnakeLin/pytorch-YOLOv4-for-Pedestrian
```
# 2. 数据准备

准备train.txt,内容是图片名和box 格式如下,其中为了方便coco评估，图片命名为**_number.jpg,如 image_001.jpg ---> image_001.xml

```
image_path1 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
image_path2 x1,y1,x2,y2,id x1,y1,x2,y2,id x1,y1,x2,y2,id ...
...
```
- image_path : 图片名
- x1,y1 : 左上角坐标
- x2,y2 : 右下角坐标
- id : 物体类别
数据集下载：Google Drive()

# 3. 参数设置
## cfg/yolov4-pedestrain.cfg
```
# Testing
#batch=1
#subdivisions=1
# Training
batch=64                //每次迭代要进行训练的图片数量 ,在一定范围内，一般来说Batch_Size越大，其确定的下降方向越准，引起的训练震荡越小。 
subdivisions=8          //源码中的图片数量int imgs = net.batch * net.subdivisions * ngpus，按subdivisions大小分批进行训练 
height=416              //输入图片高度,必须能够被32整除
width=416               //输入图片宽度,必须能够被32整除
channels=3              //输入图片通道数
momentum=0.9            //冲量
decay=0.0005            //权值衰减
angle=0                 //图片角度变化，单位为度,假如angle=5，就是生成新图片的时候随机旋转-5~5度    
saturation = 1.5        //饱和度变化大小
exposure = 1.5          //曝光变化大小
hue=.1                  //色调变化范围，tiny-yolo-voc.cfg中-0.1~0.1 
learning_rate=0.001     //学习率
burn_in=1000
max_batches = 120200    //训练次数，建议设置为classes*2000，但是不要低于4000
policy=steps            //调整学习率的策略
//根据batch_num调整学习率，若steps=100,25000,35000，则在迭代100次，25000次，35000次时学习率发生变化，该参数与policy中的steps对应
steps=40000,80000     // 一般设置为max_batch的80%与90%
scales=.1,.1             //相对于当前学习率的变化比率，累计相乘，与steps中的参数个数保持一致；

修改三处classes,分别位于970行、1058行与1146行，将其修改为自己数据集的目标数量；
classes = 2
修改三处filters,963行、1051行与1139行，将其修改为自己数据集的目标数量；
filters = 21((5+classes)*3)
```

# 4. 开始训练

```
 python train.py -l 0.002 -g 0 -pretrained yolov4.conv.137.pth -classes 2 -dir pedestrain

-l 学习率
-g gpu id
-pretrained 预训练的主干网络，从AlexeyAB给的darknet的yolov4.conv.137转换过来的
-classes 类别种类
-dir 图片所在文件夹
```


# 5. 验证

```
python video.py 2 checkpoints/Yolov4_epoch150.pth data/Pedestrians.mp4 320 320 data/pedestrain.names
python models.py 2 checkpoints/Yolov4_epoch150.pth data/people.jpg 320 320

python models.py num_classes weightfile imagepath namefile
```
# 6. 评估
```
训练150轮，map@0.5 0.61 map@0.5:0.95 0.42
```
