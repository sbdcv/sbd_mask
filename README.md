# sbd_mask
sbd_mask
这是思百达物联网科技（北京）有限公司 开源的一款轻量级cpu版实时口罩检测项目。  
检测部分网络是centerface，高效的轻量级目标检测网络  
centerface widerface测试集的精度：
|easy set |medium set |hard set|
-----|------|-----
|93.2%|92.1%|87.3%|

口罩分类网络是我们自己寻找的数据集，用mobilenetv2训练得到，以下是准确率:

|训练集|测试集|
------|------
|99%|98.5%|

速度和精度远超百度开源的轻量级口罩检测。

 速度是百度的两倍多左右
 精度也远超百度paddlehub的口罩检测，
 下面是效果对比图：
 ![image](https://github.com/sbdcv/sbd_mask/raw/master/images/1582529835.png)
 ![image](https://github.com/sbdcv/sbd_mask/raw/master/images/1582530011.png)
 ![image](https://github.com/sbdcv/sbd_mask/raw/master/images/1582529835.png)

图中左边是百度口罩检测sdk的检测结果，右图是思百达公司的口罩检测，
可以看到误检和漏检，思百达研发的都比百度的效果好。  

项目中有百度paddlehub口罩检测的代码:  
依赖项：  
pip install --upgrade paddlehub -i https://pypi.tuna.tsinghua.edu.cn/simple  
pip install paddlepaddle  
调用方法：  
python demo_baidu.py  

调用思百达的口罩检测：  
依赖项：  
python-opencv 4.x    

python demo_sbd.py  

我们还自研了轻量级人脸检测算法，精度超越centerface，会在后续继续开源出来。  
感谢大家的支持!感觉好用，还请给个star!您的支持，是我们前进的动力!
