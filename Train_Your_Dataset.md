# 1. 代码准备

github 克隆代码
```
git clone https://github.com/BigSnakeLin/pytorch-YOLOv4-for-Pedestrian
```
# 2. 数据准备

准备train.txt,内容是图片名和box 格式如下

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

